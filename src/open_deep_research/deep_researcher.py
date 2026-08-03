"""Main LangGraph implementation for Project Omega V2."""
import asyncio
import hashlib
import logging
import re
import textwrap

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage, 
    filter_messages, get_buffer_string
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    clarify_with_user_instructions, compress_research_simple_human_message,
    compress_research_system_prompt, final_report_generation_prompt,
    lead_researcher_prompt, research_system_prompt,
    transform_messages_into_research_topic_prompt, meta_learning_prompt,
    reasoning_council_prompt, meta_cognitive_router_prompt,
)
from open_deep_research.state import (
    AgentInputState, AgentState, ClarifyWithUser, ConductResearch,
    EvidenceGraphExtraction, ResearchComplete, ResearcherOutputState,
    ResearcherState, ResearchQuestion, RouterDecision, SupervisorState,
)
from open_deep_research.utils import (
    check_information_satiation, filter_and_verify_evidence, get_all_tools,
    get_api_key_for_model, get_model_token_limit, get_notes_from_tool_calls,
    get_today_str, is_token_limit_exceeded, validate_urls, think_tool,
    omega_local_memory, verify_citations_programmatically,
    calculate_epistemic_saturation, programmatic_epistemic_verification,
    _shield, groq_burst_semaphore, omega_memory,
)

configurable_model = init_chat_model(configurable_fields=("model", "max_tokens", "api_key"))

_GROQ_BURST_LIMIT = 2
_GROQ_BURST_SEMAPHORES: dict[int, asyncio.Semaphore] = {}
_COMPRESSION_TARGET_TOKENS = 800
_COMPRESSION_MAX_MESSAGES = 2
_COMPRESSION_MAX_SEGMENT_CHARS = 180
_COMPRESSION_TOKEN_SAFETY_FACTOR = 1.6


def _get_groq_burst_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _GROQ_BURST_SEMAPHORES.get(key)
    if sem is None or getattr(sem, "_loop", None) is not loop:
        sem = asyncio.Semaphore(_GROQ_BURST_LIMIT)
        _GROQ_BURST_SEMAPHORES[key] = sem
    return sem

def add_targeted_research_nodes(evidence_graph: list, research_plan: list) -> list:
    plan = list(research_plan) if research_plan else []
    existing = {node.get("node_id") for node in plan if isinstance(node, dict)}
    targets = []
    for idx, node in enumerate((evidence_graph or [])[:3], start=1):
        claim = str(getattr(node, "claim", "")).strip()
        if not claim:
            continue
        targets.append({
            "node_id": f"FB_{idx}",
            "topic": f"Verify, stress-test, and resolve: {claim[:180]}",
            "depends_on": [],
        })
    if not targets:
        targets = [{
            "node_id": "FB_1",
            "topic": "Resolve contradictions and verify the weakest evidence nodes.",
            "depends_on": [],
        }]
    for node in targets:
        if node["node_id"] not in existing:
            plan.append(node)
    return plan

def generate_argus_view(nodes: list) -> str:
    if not nodes:
        return "No structured evidence gathered."
    support_counts = {n.citation_index: 0 for n in nodes}
    for n in nodes:
        for s in getattr(n, "supports", []):
            if s in support_counts:
                support_counts[s] += 1
    foundational = [n for n in nodes if support_counts.get(n.citation_index, 0) >= 2]
    contradicted = [n for n in nodes if getattr(n, "contradicts", [])]
    core = [n for n in nodes if n not in foundational and n not in contradicted][:10]
    
    view = "### ARGUS TOPOLOGICAL VIEW\n"
    if foundational:
        view += "Foundational Consensus:\n"
        for n in foundational[:5]:
            view += f"- [{n.citation_index}] {n.claim} (Supported by {support_counts[n.citation_index]})\n"
    if contradicted:
        view += "\nActive Dissent:\n"
        for n in contradicted[:5]:
            view += f"- [{n.citation_index}] {n.claim} (Contradicts: {', '.join(str(c) for c in n.contradicts)})\n"
    if core:
        view += "\nPeripheral Claims:\n"
        for n in core:
            view += f"- [{n.citation_index}] {n.claim}\n"
    return view

def _erc_normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _erc_source_family(url):
    if not url:
        return ""
    try:
        host = str(url).split("://", 1)[-1].split("/", 1)[0].lower().replace("www.", "")
    except Exception:
        host = str(url).lower()
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _erc_node_signature(node):
    claim = _erc_normalize_text(getattr(node, "claim", ""))
    url = _erc_normalize_text(getattr(node, "url", ""))
    title = _erc_normalize_text(getattr(node, "title", ""))
    date_published = _erc_normalize_text(getattr(node, "date_published", ""))
    supports = ",".join(str(s) for s in sorted(set(getattr(node, "supports", []) or [])))
    contradicts = ",".join(str(c) for c in sorted(set(getattr(node, "contradicts", []) or [])))
    return "|".join([claim, url, title, date_published, supports, contradicts])


def _erc_build_snapshot(state, evidence_graph=None, research_plan=None, completed_nodes=None):
    current_evidence = list(evidence_graph if evidence_graph is not None else state.get("evidence_graph", []))
    current_plan = list(research_plan if research_plan is not None else state.get("research_plan", []))
    current_completed = list(dict.fromkeys(list(state.get("completed_nodes", [])) + list(completed_nodes or [])))

    urls = []
    for node in current_evidence:
        url = str(getattr(node, "url", "") or "").strip()
        if url and url not in urls:
            urls.append(url)

    source_family_coverage = {}
    for url in urls:
        family = _erc_source_family(url)
        if not family:
            continue
        source_family_coverage[family] = source_family_coverage.get(family, 0) + 1

    open_questions = []
    completed_set = set(current_completed)
    for item in current_plan:
        if isinstance(item, dict):
            node_id = str(item.get("node_id", "") or "").strip()
            topic = str(item.get("topic", "") or "").strip()
        else:
            node_id = str(getattr(item, "node_id", "") or "").strip()
            topic = str(getattr(item, "topic", "") or "").strip()
        if node_id and node_id not in completed_set:
            open_questions.append(f"{node_id}: {topic}")

    unresolved_contradictions = []
    for node in current_evidence:
        contrad = list(getattr(node, "contradicts", []) or [])
        if contrad:
            citation = getattr(node, "citation_index", 0)
            claim = str(getattr(node, "claim", "") or "").strip()
            unresolved_contradictions.append(f"[{citation}] {claim}")

    evidence_count = len(current_evidence)
    completed_count = len(current_completed)
    plan_count = len(current_plan)

    fingerprint_raw = "||".join([
        "E:" + ";".join(sorted(_erc_node_signature(n) for n in current_evidence)),
        "P:" + ";".join(sorted(
            str(item.get("node_id", "") if isinstance(item, dict) else getattr(item, "node_id", "")).strip().lower()
            + ":" +
            str(item.get("topic", "") if isinstance(item, dict) else getattr(item, "topic", "")).strip().lower()
            for item in current_plan
        )),
        "C:" + ";".join(sorted(str(x).strip().lower() for x in current_completed if str(x).strip())),
        "F:" + ";".join(f"{k}:{v}" for k, v in sorted(source_family_coverage.items())),
        "O:" + ";".join(sorted(_erc_normalize_text(q) for q in open_questions)),
        "X:" + ";".join(sorted(_erc_normalize_text(x) for x in unresolved_contradictions)),
        "N:" + str(evidence_count),
        "M:" + str(completed_count),
        "R:" + str(plan_count),
    ])
    current_fp = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()
    previous_fp = str(state.get("erc_frontier_fingerprint", "") or "")
    no_progress_count = int(state.get("erc_no_progress_count", 0) or 0)
    if previous_fp and current_fp == previous_fp:
        no_progress_count += 1
    else:
        no_progress_count = 0

    return {
        "erc_frontier_fingerprint": current_fp,
        "erc_previous_fingerprint": previous_fp,
        "erc_no_progress_count": no_progress_count,
        "erc_last_evidence_count": evidence_count,
        "erc_last_completed_count": completed_count,
        "erc_open_questions": open_questions,
        "erc_unresolved_contradictions": unresolved_contradictions,
        "erc_source_family_coverage": source_family_coverage,
    }


def _erc_frontier_open(snapshot):
    return bool(snapshot.get("erc_open_questions") or snapshot.get("erc_unresolved_contradictions"))


def _erc_frontier_note(snapshot) -> str:
    families = snapshot.get("erc_source_family_coverage", {}) or {}
    return (
        "ERC frontier status: "
        f"open_questions={len(snapshot.get('erc_open_questions', []) or [])}, "
        f"unresolved_contradictions={len(snapshot.get('erc_unresolved_contradictions', []) or [])}, "
        f"source_families={len(families)}, "
        f"stagnation={snapshot.get('erc_no_progress_count', 0)}."
    )

def _erc_plan_item_to_dict(item):
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        try:
            return item.model_dump()
        except Exception:
            pass
    if hasattr(item, "dict"):
        try:
            return item.dict()
        except Exception:
            pass
    try:
        return dict(item)
    except Exception:
        return {"topic": str(item)}


async def safe_llm_invoke(model, messages):
    try:
        sem = globals().get('groq_burst_semaphore')
        if sem is None and '_get_groq_burst_semaphore' in globals():
            sem = _get_groq_burst_semaphore()
        if sem is None:
            sem = asyncio.Semaphore(3)
        async with sem:
            for attempt in range(3):
                try: 
                    return await model.ainvoke(messages)
                except Exception as e:
                    err = str(e).lower()
                    if "rate limit" in err or "429" in err or "413" in err or "resource_exhausted" in err or "timeout" in err or "413" in err:
                        new_key = _shield.get_key()
                        model = model.with_config({"api_key": new_key})
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    if attempt == 2: raise RuntimeError("[EPISTEMIC FLAG]: LLM infrastructure constraint.")
            raise RuntimeError("[EPISTEMIC FLAG]: LLM infrastructure constraint.")
    except Exception as e:
        logging.error(f"safe_llm_invoke fatal: {e}")
        raise e


async def clarify_with_user(state: AgentState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        if not configurable.allow_clarification:
            return Command(goto="write_research_brief")
        mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        cm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(ClarifyWithUser)
        prompt = clarify_with_user_instructions.format(messages=get_buffer_string(state["messages"]), date=get_today_str())
        response = await safe_llm_invoke(cm, [HumanMessage(content=prompt)])
        if response.need_clarification:
            return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})
    except Exception as e:
        logging.error(f"clarify_with_user failed: {e}")
        return Command(goto="write_research_brief")

async def write_research_brief(state: AgentState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(ResearchQuestion)
        prompt = transform_messages_into_research_topic_prompt.format(messages=get_buffer_string(state.get("messages", [])), date=get_today_str())
        response = await safe_llm_invoke(rm, [HumanMessage(content=prompt)])
        mem_ctx = omega_memory.get_context_prompt()
        sup_sys = lead_researcher_prompt.format(
            date=get_today_str(),
            mcp_prompt=configurable.mcp_prompt or "",
            max_concurrent_research_units=configurable.max_concurrent_research_units,
            max_researcher_iterations=configurable.max_researcher_iterations,
            temporal_intent=getattr(response, "temporal_intent", "Current"),
            complexity_tier="Pending",
            lessons_learned=mem_ctx,
            hard_constraints=getattr(response, "hard_constraints", []),
            memory_context=mem_ctx,
        )
        return Command(goto="meta_cognitive_router", update={
            "research_brief": response.research_brief, 
            "temporal_intent": getattr(response, "temporal_intent", "Current"), 
            "hard_constraints": getattr(response, "hard_constraints", []), 
            "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=response.research_brief)]}
        })
    except Exception as e:
        logging.error(f"write_research_brief failed: {e}")
        return Command(goto=END)

async def meta_cognitive_router(state: AgentState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        mc = {"model": configurable.research_model, "max_tokens": 4096, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(RouterDecision)
        mem_ctx = omega_memory.get_context_prompt()
        prompt = meta_cognitive_router_prompt.format(research_brief=state.get("research_brief", ""), date=get_today_str(), memory_context=mem_ctx)
        response = await safe_llm_invoke(rm, [HumanMessage(content=prompt)])
        sup_sys = lead_researcher_prompt.format(
            date=get_today_str(),
            mcp_prompt=configurable.mcp_prompt or "",
            max_concurrent_research_units=response.dynamic_research_units,
            max_researcher_iterations=response.dynamic_tool_budget,
            complexity_tier=response.complexity_tier,
            temporal_intent=state.get("temporal_intent", "Current"),
            lessons_learned=mem_ctx,
            hard_constraints=state.get("hard_constraints", []),
            memory_context=mem_ctx,
        )
        plan_dicts = [_erc_plan_item_to_dict(n) for n in (response.research_plan or [])]
        return Command(goto="research_supervisor", update={
            "query_paradigm": response.query_paradigm, 
            "complexity_tier": response.complexity_tier, 
            "dynamic_tool_budget": response.dynamic_tool_budget, 
            "dynamic_research_units": response.dynamic_research_units, 
            "research_plan": plan_dicts, 
            "completed_nodes": [], 
            "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=state.get("research_brief", ""))]}
        })
    except Exception as e:
        logging.error(f"meta_cognitive_router failed: {e}")
        return Command(goto=END)
async def supervisor(state: SupervisorState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        mc = {
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.research_model, config),
            "tags": ["langsmith:nostream"],
        }
        rm = configurable_model.bind_tools(
            [ConductResearch, ResearchComplete, think_tool]
        ).with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        ).with_config(mc)

        erc_snapshot = _erc_build_snapshot(state)
        saturation = calculate_epistemic_saturation(
            state.get("evidence_graph", []),
            state.get("research_plan", []),
        )

        sup_msgs = list(state.get("supervisor_messages", []))
        core_msgs = [
            m for m in sup_msgs
            if isinstance(m, (SystemMessage, HumanMessage))
            and "DAG_STATUS" not in str(getattr(m, "content", ""))
        ]
        recent_msgs = [m for m in sup_msgs if not isinstance(m, (SystemMessage, HumanMessage))][-4:]
        sup_msgs = core_msgs + recent_msgs

        plan = state.get("research_plan", [])
        completed = state.get("completed_nodes", [])

        progress_note = (
            f"\n<ERC_STATUS>\n"
            f"{_erc_frontier_note(erc_snapshot)}\n"
            f"Coverage(saturation)={saturation:.2f}\n"
            f"Plan size={len(plan)}\n"
            f"Completed={len(completed)}\n"
            f"</ERC_STATUS>"
        )
        sup_msgs.append(SystemMessage(content=progress_note))

        if saturation >= 0.85:
            sup_msgs.append(SystemMessage(content=(
                "Coverage is high, but do not stop. "
                "Continue searching for missing evidence, contradictions, and adjacent source families. "
                "Do not emit ResearchComplete until the frontier is actually closed."
            )))

        if erc_snapshot["erc_no_progress_count"] >= 2:
            sup_msgs.append(SystemMessage(content=(
                "ERC: the current frontier appears stagnant. "
                "Diversify the search, target unresolved questions, and seek alternate source families."
            )))

        response = await safe_llm_invoke(rm, sup_msgs)
        update_payload = {
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        }
        update_payload.update(erc_snapshot)
        return Command(goto="supervisor_tools", update=update_payload)
    except Exception as e:
        logging.error(f"supervisor failed: {e}")
        return Command(goto=END)

async def supervisor_tools(state: SupervisorState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        snapshot = _erc_build_snapshot(state)
        sup_msgs = state.get("supervisor_messages", [])
        if not sup_msgs:
            if _erc_frontier_open(snapshot):
                return Command(goto="supervisor", update=snapshot)
            return Command(goto=END)

        iters = state.get("research_iterations", 0)
        last_msg = sup_msgs[-1]
        tool_calls = list(getattr(last_msg, "tool_calls", []) or [])
        requested_completion = any(tc.get("name") == "ResearchComplete" for tc in tool_calls)

        if iters > configurable.max_researcher_iterations:
            if _erc_frontier_open(snapshot):
                broadened_plan = add_targeted_research_nodes(
                    state.get("evidence_graph", []),
                    state.get("research_plan", []),
                )
                return Command(goto="supervisor", update={
                    **snapshot,
                    "research_plan": broadened_plan,
                    "complexity_tier": "Complex",
                    "supervisor_messages": [SystemMessage(content=(
                        "ERC: iteration budget hit, but the frontier is still open. "
                        "Broaden the search and continue."
                    ))],
                })
            return Command(goto=END, update={"notes": get_notes_from_tool_calls(sup_msgs), "research_brief": state.get("research_brief", "")})

        if not tool_calls:
            if _erc_frontier_open(snapshot):
                return Command(goto="supervisor", update={
                    **snapshot,
                    "supervisor_messages": [SystemMessage(content=(
                        "ERC: no tool calls were produced. Continue by issuing ConductResearch "
                        "against the remaining open frontier and alternate source families."
                    ))],
                })
            return Command(goto=END, update={"notes": get_notes_from_tool_calls(sup_msgs), "research_brief": state.get("research_brief", "")})

        all_tool_msgs = []
        update_payload = {"supervisor_messages": []}
        conduct_calls = [t for t in tool_calls if t.get("name") == "ConductResearch"]

        if conduct_calls:
            allowed = conduct_calls[:configurable.max_concurrent_research_units]
            tasks = []
            for tc in allowed:
                base_topic = tc["args"]["research_topic"]
                invariant_payload = (
                    f"{base_topic}\n\n[INVARIANT CONSTRAINTS]\n"
                    f"Temporal: {state.get('temporal_intent')}\n"
                    f"Hard Constraints: {state.get('hard_constraints')}"
                )
                tasks.append(
                    researcher_subgraph.ainvoke(
                        {
                            "researcher_messages": [HumanMessage(content=invariant_payload)],
                            "research_topic": invariant_payload,
                        },
                        config,
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            vfs_update, agg_graph, newly_completed = {}, [], []

            for obs, tc in zip(results, allowed):
                if isinstance(obs, Exception):
                    all_tool_msgs.append(
                        ToolMessage(
                            content=f"[FALLBACK] Researcher failed: {str(obs)}",
                            name=tc["name"],
                            tool_call_id=tc["id"],
                        )
                    )
                    continue

                art_id = str(obs.get("artifact_id", tc["id"]))
                vfs_update[art_id] = obs.get("compressed_research", "")
                all_tool_msgs.append(
                    ToolMessage(
                        content=(
                            f"ARTIFACT ID: {art_id}\n"
                            f"SUMMARY: {obs.get('executive_summary', 'Done')}\n"
                            f"[VFS STORED]"
                        ),
                        name=tc["name"],
                        tool_call_id=tc["id"],
                    )
                )
                agg_graph.extend(obs.get("evidence_graph", []))
                if tc["args"].get("node_id"):
                    newly_completed.append(tc["args"].get("node_id"))

            if vfs_update:
                update_payload["virtual_filesystem"] = vfs_update
            if agg_graph:
                update_payload["evidence_graph"] = agg_graph
            if newly_completed:
                update_payload["completed_nodes"] = list(set(state.get("completed_nodes", [])).union(set(newly_completed)))

            update_payload.update(
                _erc_build_snapshot(
                    state,
                    evidence_graph=list(state.get("evidence_graph", [])) + agg_graph,
                    research_plan=state.get("research_plan", []),
                    completed_nodes=newly_completed,
                )
            )

            if requested_completion and _erc_frontier_open(update_payload):
                all_tool_msgs.append(
                    ToolMessage(
                        content=(
                            "[IGNORED] Premature completion request rejected. "
                            "The frontier is still open, so continue researching."
                        ),
                        name="ResearchComplete",
                        tool_call_id=tool_calls[0].get("id", "completion_guard"),
                    )
                )

            if update_payload.get("erc_no_progress_count", 0) >= 2 or not agg_graph:
                update_payload["research_plan"] = add_targeted_research_nodes(
                    list(state.get("evidence_graph", [])) + agg_graph,
                    state.get("research_plan", []),
                )
                update_payload["complexity_tier"] = "Complex"

        else:
            update_payload.update(snapshot)

        update_payload["supervisor_messages"] = all_tool_msgs
        return Command(goto="supervisor", update=update_payload)
    except Exception as e:
        logging.error(f"supervisor_tools failed: {e}")
        return Command(goto=END)

supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()

async def execute_tool_safely(tool, args, config):
    for attempt in range(3):
        try:
            return await tool.ainvoke(args, config)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e).lower():
                await asyncio.sleep(2 ** attempt)
            else:
                return f"[TOOL FALLBACK]: {getattr(tool, 'name', 'unknown')} failed."
    return "[TOOL FALLBACK]: Max retries exceeded."

async def researcher(state: ResearcherState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        tools = await get_all_tools(config)
        mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        prompt = research_system_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_str(), temporal_intent=state.get("temporal_intent", "Current"), hard_constraints=state.get("hard_constraints", []), memory_context=omega_memory.get_context_prompt())
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).bind_tools(tools)
        r_msgs = state.get("researcher_messages", [])
        core_r = [m for m in r_msgs if isinstance(m, (SystemMessage, HumanMessage))]
        recent_r = [m for m in r_msgs if not isinstance(m, (SystemMessage, HumanMessage))][-6:]
        msgs = [SystemMessage(content=prompt)] + core_r + recent_r
        response = await safe_llm_invoke(rm, msgs)
        return Command(goto="researcher_tools", update={"researcher_messages": [response], "tool_call_iterations": state.get("tool_call_iterations", 0) + 1})
    except Exception as e:
        logging.error(f"researcher failed: {e}")
        return Command(goto="compress_research")

async def researcher_tools(state: ResearcherState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        r_msgs = state.get("researcher_messages", [])
        if not r_msgs:
            return Command(goto="compress_research")
        last_msg = r_msgs[-1]
        if not getattr(last_msg, "tool_calls", None):
            return Command(goto="compress_research")
        tools = await get_all_tools(config)
        tools_by_name = {t.name: t for t in tools if hasattr(t, "name")}
        obs = await asyncio.gather(*[execute_tool_safely(tools_by_name[tc["name"]], tc["args"], config) for tc in last_msg.tool_calls if tc["name"] in tools_by_name])
        tool_outputs = [ToolMessage(content=o, name=tc["name"], tool_call_id=tc["id"]) for o, tc in zip(obs, last_msg.tool_calls) if tc["name"] in tools_by_name]
        new_claims = [o for o in obs if isinstance(o, str)]
        existing_context = [m.content for m in r_msgs if hasattr(m, "content") and isinstance(m.content, str)]
        if check_information_satiation(new_claims, existing_context) or state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls:
            return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
        return Command(goto="researcher", update={"researcher_messages": tool_outputs})
    except Exception as e:
        logging.error(f"researcher_tools failed: {e}")
        return Command(goto="compress_research")

def _message_text(msg):
    return str(getattr(msg, "content", "") or "")


def _estimate_message_tokens(text: str) -> int:
    return max(1, (len(str(text or "")) + 3) // 4)


def _estimate_compression_prompt_tokens(chunk_messages) -> int:
    system_text = compress_research_system_prompt.format(date=get_today_str())
    human_text = compress_research_simple_human_message
    total = _estimate_message_tokens(system_text) + _estimate_message_tokens(human_text)
    for msg in chunk_messages:
        total += _estimate_message_tokens(_message_text(msg)) + 25
    return int((total + 24) * _COMPRESSION_TOKEN_SAFETY_FACTOR)


def _split_text_preserving_structure(text: str, max_chars: int = 1200) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts = []
    for paragraph in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if len(paragraph) <= max_chars:
            parts.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        bucket = []
        bucket_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if bucket:
                    parts.append(" ".join(bucket).strip())
                    bucket = []
                    bucket_len = 0
                wrapped = textwrap.wrap(
                    sentence,
                    width=max_chars,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                parts.extend([w.strip() for w in wrapped if w.strip()])
                continue

            next_len = bucket_len + len(sentence) + (1 if bucket else 0)
            if next_len <= max_chars:
                bucket.append(sentence)
                bucket_len = next_len
            else:
                if bucket:
                    parts.append(" ".join(bucket).strip())
                bucket = [sentence]
                bucket_len = len(sentence)

        if bucket:
            parts.append(" ".join(bucket).strip())

    if not parts:
        wrapped = textwrap.wrap(
            text,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )
        return [w.strip() for w in wrapped if w.strip()] or [text[:max_chars]]

    return [p for p in parts if p.strip()]


def _clone_message_with_content(msg, content):
    if isinstance(msg, HumanMessage):
        return HumanMessage(content=content)
    if isinstance(msg, AIMessage):
        return AIMessage(content=content)
    if isinstance(msg, SystemMessage):
        return SystemMessage(content=content)
    if isinstance(msg, ToolMessage):
        kwargs = {"content": content, "tool_call_id": getattr(msg, "tool_call_id", "split_tool_call")}
        name = getattr(msg, "name", None)
        if name:
            kwargs["name"] = name
        try:
            return ToolMessage(**kwargs)
        except Exception:
            return HumanMessage(content=content)
    try:
        return type(msg)(content=content)
    except Exception:
        return HumanMessage(content=content)


def _expand_messages_for_compression(messages, max_segment_chars: int = 1200):
    expanded = []
    for msg in messages:
        text = _message_text(msg).strip()
        if not text:
            continue
        parts = _split_text_preserving_structure(text, max_chars=max_segment_chars)
        if len(parts) <= 1:
            expanded.append(msg)
            continue
        total = len(parts)
        for idx, part in enumerate(parts, start=1):
            prefix = f"[PART {idx}/{total}] "
            expanded.append(_clone_message_with_content(msg, prefix + part))
    return expanded

def _chunk_messages_for_compression(
    messages,
    target_tokens=_COMPRESSION_TARGET_TOKENS,
    max_messages=_COMPRESSION_MAX_MESSAGES,
    max_segment_chars=_COMPRESSION_MAX_SEGMENT_CHARS,
):
    prepared = _expand_messages_for_compression(
        messages,
        max_segment_chars=max_segment_chars,
    )
    chunks = []
    current = []

    for msg in prepared:
        candidate = current + [msg]
        candidate_tokens = _estimate_compression_prompt_tokens(candidate)

        if current and (candidate_tokens > target_tokens or len(candidate) > max_messages):
            chunks.append(current)
            current = [msg]
            continue

        if not current and candidate_tokens > target_tokens and len(prepared) > 1:
            chunks.append([msg])
            current = []
            continue

        current = candidate

    if current:
        chunks.append(current)

    return chunks


def _evidence_key(node):
    return (
        str(getattr(node, "url", "") or "").strip(),
        str(getattr(node, "claim", "") or "").strip().lower(),
        str(getattr(node, "date_published", "") or "").strip(),
    )


def _quality_score(node):
    claim = str(getattr(node, "claim", "") or "")
    title = str(getattr(node, "title", "") or "")
    snippet = str(getattr(node, "snippet", "") or "")
    url = str(getattr(node, "url", "") or "")
    return (len(claim) * 3) + len(title) + len(snippet) + len(url)


def _merge_evidence_nodes(node_groups):
    merged = {}
    order = []

    for group in node_groups:
        for node in group:
            if node is None:
                continue

            key = _evidence_key(node)
            if not key[0] and not key[1]:
                continue

            if key not in merged:
                merged[key] = node
                order.append(key)
                continue

            existing = merged[key]
            support_values = list(
                dict.fromkeys(
                    list(getattr(existing, "supports", []) or [])
                    + list(getattr(node, "supports", []) or [])
                )
            )
            contradict_values = list(
                dict.fromkeys(
                    list(getattr(existing, "contradicts", []) or [])
                    + list(getattr(node, "contradicts", []) or [])
                )
            )
            chosen = node if _quality_score(node) >= _quality_score(existing) else existing

            if hasattr(chosen, "model_copy"):
                merged[key] = chosen.model_copy(
                    update={
                        "supports": support_values,
                        "contradicts": contradict_values,
                    }
                )
            else:
                merged[key] = chosen

    return [merged[key] for key in order]



def _render_evidence_graph(nodes):
    if not nodes:
        return "No structured evidence gathered."

    lines = ["Extracted Evidence Graph:"]
    for i, node in enumerate(nodes):
        claim = str(getattr(node, "claim", "") or "")
        title = str(getattr(node, "title", "") or "")
        url = str(getattr(node, "url", "") or "")
        date_published = getattr(node, "date_published", None)
        supports = getattr(node, "supports", [])
        contradicts = getattr(node, "contradicts", [])

        lines.append(f"Fact {i + 1}: {claim}")
        lines.append(f"Source: {title} ({url})")
        if date_published:
            lines.append(f"Date: {date_published}")
        lines.append(f"Supports: {supports} | Contradicts: {contradicts}")
        lines.append("")

    return "\n".join(lines).strip()



async def _compress_message_chunk(sm, chunk_messages, configurable, depth=0, target_tokens=_COMPRESSION_TARGET_TOKENS):
    if not chunk_messages:
        return [], ""

    prompt_tokens = _estimate_compression_prompt_tokens(chunk_messages)
    if prompt_tokens > target_tokens and depth < 5 and len(chunk_messages) > 1:
        mid = max(1, len(chunk_messages) // 2)
        left_nodes, left_notes = await _compress_message_chunk(
            sm, chunk_messages[:mid], configurable, depth + 1, target_tokens=target_tokens
        )
        right_nodes, right_notes = await _compress_message_chunk(
            sm, chunk_messages[mid:], configurable, depth + 1, target_tokens=target_tokens
        )
        combined_notes = "\n\n".join([n for n in [left_notes, right_notes] if n])
        return left_nodes + right_nodes, combined_notes

    if prompt_tokens > target_tokens and depth < 5 and len(chunk_messages) == 1:
        expanded = _expand_messages_for_compression(chunk_messages, max_segment_chars=max(80, _COMPRESSION_MAX_SEGMENT_CHARS // 2))
        if len(expanded) > len(chunk_messages):
            return await _compress_message_chunk(
                sm, expanded, configurable, depth + 1, target_tokens=target_tokens
            )

    prompt_messages = [
        SystemMessage(content=compress_research_system_prompt.format(date=get_today_str())),
        *chunk_messages,
        HumanMessage(content=compress_research_simple_human_message),
    ]

    try:
        response = await sm.ainvoke(prompt_messages)
        raw_notes = "\n".join(
            str(m.content) for m in filter_messages(chunk_messages, include_types=["tool", "ai"])
        )
        return response.nodes, raw_notes
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.compression_model) and depth < 5:
            if len(chunk_messages) > 1:
                mid = max(1, len(chunk_messages) // 2)
                left_nodes, left_notes = await _compress_message_chunk(
                    sm, chunk_messages[:mid], configurable, depth + 1, target_tokens=target_tokens
                )
                right_nodes, right_notes = await _compress_message_chunk(
                    sm, chunk_messages[mid:], configurable, depth + 1, target_tokens=target_tokens
                )
                combined_notes = "\n\n".join([n for n in [left_notes, right_notes] if n])
                return left_nodes + right_nodes, combined_notes

            split_single = _expand_messages_for_compression(chunk_messages, max_segment_chars=250)
            if len(split_single) > len(chunk_messages):
                return await _compress_message_chunk(
                    sm, split_single, configurable, depth + 1, target_tokens=target_tokens
                )
        raise



def _truncate_messages_for_compression(msgs, max_chars=12000):
    from langchain_core.messages import ToolMessage
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    other_msgs = [m for m in msgs if not isinstance(m, ToolMessage)]
    
    total_other_chars = sum(len(str(m.content)) for m in other_msgs)
    remaining_chars = max(2000, max_chars - total_other_chars)
    per_tool_limit = max(500, remaining_chars // max(len(tool_msgs), 1))
    
    truncated_tools = []
    for m in tool_msgs:
        content = str(m.content)
        if len(content) > per_tool_limit:
            half = per_tool_limit // 2
            new_content = content[:half] + "\n...[TRUNCATED FOR COMPRESSION]...\n" + content[-half:]
            truncated_tools.append(ToolMessage(content=new_content, tool_call_id=m.tool_call_id, name=getattr(m, "name", "tool")))
        else:
            truncated_tools.append(m)
            
    return other_msgs + truncated_tools


def _truncate_for_compression(msgs, max_chars=8000):
    from langchain_core.messages import ToolMessage
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    other_msgs = [m for m in msgs if not isinstance(m, ToolMessage)]
    remaining = max(2000, max_chars - sum(len(str(m.content)) for m in other_msgs))
    per_tool = max(400, remaining // max(len(tool_msgs), 1))
    out = []
    for m in tool_msgs:
        c = str(m.content)
        if len(c) > per_tool:
            half = per_tool // 2
            c = c[:half] + "\n[TRUNCATED]\n" + c[-half:]
        out.append(ToolMessage(content=c, tool_call_id=m.tool_call_id, name=getattr(m, "name", "tool")))
    return other_msgs + out

async def compress_research(state: ResearcherState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        mc = {
            "model": configurable.compression_model,
            "max_tokens": configurable.compression_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.compression_model, config),
            "tags": ["langsmith:nostream"],
        }
        sm = init_chat_model(
            model=mc["model"],
            max_tokens=mc["max_tokens"],
            api_key=mc["api_key"],
        ).with_structured_output(EvidenceGraphExtraction)

        source_messages = list(state.get("researcher_messages", []))
        if not source_messages:
            return {
                "compressed_research": "Error",
                "raw_notes": [""],
                "evidence_graph": [],
                "artifact_id": "error",
                "executive_summary": "Failed",
            }

        chunks = _chunk_messages_for_compression(
            source_messages,
            target_tokens=_COMPRESSION_TARGET_TOKENS,
            max_messages=_COMPRESSION_MAX_MESSAGES,
            max_segment_chars=_COMPRESSION_MAX_SEGMENT_CHARS,
        )
        if not chunks:
            return {
                "compressed_research": "Error",
                "raw_notes": [""],
                "evidence_graph": [],
                "artifact_id": "error",
                "executive_summary": "Failed",
            }

        all_node_groups = []
        raw_notes_parts = []

        for chunk in chunks:
            chunk_nodes, chunk_raw_notes = await _compress_message_chunk(sm, chunk, configurable)
            all_node_groups.append(chunk_nodes)
            if chunk_raw_notes:
                raw_notes_parts.append(chunk_raw_notes)

        merged_nodes = _merge_evidence_nodes(all_node_groups)

        for node in merged_nodes:
            try:
                omega_local_memory.store(getattr(node, "claim", ""), getattr(node, "url", ""))
            except Exception:
                pass

        raw_notes = "\n\n".join(raw_notes_parts)
        art_id = hashlib.md5(raw_notes.encode()).hexdigest()[:8]
        readable = (
            f"Extracted Evidence Graph ({len(merged_nodes)} atomic facts from {len(chunks)} chunk(s)):\n"
            + _render_evidence_graph(merged_nodes)
        )
        executive_summary = readable[:800]

        return {
            "compressed_research": readable,
            "raw_notes": [raw_notes],
            "evidence_graph": merged_nodes,
            "artifact_id": art_id,
            "executive_summary": executive_summary,
        }
    except Exception as e:
        logging.error(f"compress_research failed: {e}")
        return {
            "compressed_research": "Error",
            "raw_notes": [],
            "evidence_graph": [],
            "artifact_id": "error",
            "executive_summary": "Failed",
        }


researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def reasoning_council(state: AgentState, config: RunnableConfig):
    try:
        tier = state.get("complexity_tier", "Medium")
        if tier in ["Simple", "Medium"]:
            return Command(goto="adversarial_verification", update={"master_synthesis": "Standard inductive synthesis applied."})
        configurable = Configuration.from_runnable_config(config)
        raw_notes = "\n".join(state.get("notes", []))
        argus = generate_argus_view(state.get("evidence_graph", []))
        findings = argus + "\n\n### Raw Notes\n" + raw_notes[:8000]
        brief = state.get("research_brief", "")
        paradigms = ["Deductive", "Inductive", "Abductive", "Analogical", "Probabilistic"]
        async def run_p(p):
            mc = {"model": configurable.research_model, "max_tokens": 2048, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
            try:
                prompt = reasoning_council_prompt.format(paradigm=p, brief=brief, findings=findings[:15000], master_synthesis="", topic=brief, context="", perspectives=p)
                res = await safe_llm_invoke(init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]), [HumanMessage(content=prompt)])
                return f"### {p} Perspective\n{res.content}"
            except Exception:
                return f"### {p} Perspective\n[EPISTEMIC FLAG]: Skipped."
        results = await asyncio.gather(*[run_p(p) for p in paradigms])
        return Command(goto="adversarial_verification", update={"master_synthesis": "\n\n".join(results)})
    except Exception as e:
        logging.error(f"reasoning_council failed: {e}")
        return Command(goto="adversarial_verification", update={"master_synthesis": "Council failed."})

async def adversarial_verification(state: AgentState, config: RunnableConfig):
    try:
        evidence = state.get("evidence_graph", [])
        temporal = state.get("temporal_intent", "Current")
        verification_results = programmatic_epistemic_verification(evidence, temporal)
        argus = generate_argus_view(evidence) if evidence else ""
        raw = state.get("master_synthesis", "")
        return Command(goto="final_report_generation", update={
            "red_team_findings": verification_results["red_team_findings"],
            "devils_advocate_critique": verification_results["devils_advocate_critique"],
            "consensus_report": verification_results["consensus_report"] + "\n\n### Argus View\n" + argus + "\n\n### Council Synthesis\n" + raw[:3000],
            "confidence_score": verification_results["confidence_score"],
        })
    except Exception as e:
        logging.error(f"adversarial_verification failed: {e}")
        return Command(goto="final_report_generation", update={"confidence_score": 0.5})

async def final_report_generation(state: AgentState, config: RunnableConfig):
    snapshot = _erc_build_snapshot(state)
    if _erc_frontier_open(snapshot):
        return Command(
            goto="research_supervisor",
            update={
                **snapshot,
                "research_plan": add_targeted_research_nodes(
                    state.get("evidence_graph", []),
                    state.get("research_plan", []),
                ),
                "complexity_tier": "Complex",
            },
        )
    try:
        raw_ev = state.get("evidence_graph", [])
        v_notes = []
        if raw_ev:
            urls = [getattr(n, "url", "") for n in raw_ev if getattr(n, "url", "")]
            if urls:
                try:
                    health = await validate_urls(urls)
                    raw_ev = [n for n in raw_ev if health.get(getattr(n, "url", ""), False)]
                except Exception:
                    pass
        verified = filter_and_verify_evidence(raw_ev, temporal_intent=state.get("temporal_intent", "Current"))
        verified = await verify_citations_programmatically(verified)
        for n in verified:
            d = str(n.date_published) if getattr(n, "date_published", None) else "Unknown"
            v_notes.append(f"Fact: {getattr(n, 'claim', '')}\nSource: {getattr(n, 'title', '')} ({getattr(n, 'url', '')})\nDate: {d}")
        notes = state.get("notes", []) + v_notes
        cleared = {"notes": {"type": "override", "value": []}}
        vfs = state.get("virtual_filesystem", {})
        vfs_evidence = "\n\n".join([f"### VFS Artifact {k}\n{v}" for k, v in vfs.items()])
        findings = "\n".join(notes) + "\n\n" + vfs_evidence
        configurable = Configuration.from_runnable_config(config)
        wc = {"model": configurable.final_report_model, "max_tokens": configurable.final_report_model_max_tokens, "api_key": get_api_key_for_model(configurable.final_report_model, config), "tags": ["langsmith:nostream"]}
        retries, limit = 0, None
        while retries <= 3:
            try:
                prompt = final_report_generation_prompt.format(
                    research_brief=state.get("research_brief", ""), messages=get_buffer_string(state.get("messages", [])),
                    findings=findings, date=get_today_str(), master_synthesis=state.get("master_synthesis", "Standard synthesis."),
                    consensus_report=state.get("consensus_report", "None."), confidence_score=state.get("confidence_score", 0.8),
                    query_paradigm=state.get("query_paradigm", "General"), topic=state.get("research_brief", ""),
                    confidence=state.get("confidence_score", 0.8), resolved_issues=""
                )
                rep = await init_chat_model(model=wc["model"], max_tokens=wc["max_tokens"], api_key=wc["api_key"]).ainvoke([HumanMessage(content=prompt)])
                return {"final_report": rep.content, "messages": [rep], **cleared}
            except Exception as e:
                if is_token_limit_exceeded(e, configurable.final_report_model):
                    retries += 1
                    if retries == 1:
                        tl = get_model_token_limit(configurable.final_report_model)
                        if not tl:
                            return {"final_report": "Error: Token limit.", "messages": [AIMessage(content="Failed")], **cleared}
                        limit = tl * 4
                    else:
                        limit = int(limit * 0.9)
                    findings = findings[:limit]
                else:
                    return {"final_report": f"Error: {str(e)}", "messages": [AIMessage(content="Failed")], **cleared}
        return {"final_report": "Error: Max retries.", "messages": [AIMessage(content="Failed")], **cleared}
    except Exception as e:
        logging.error(f"final_report_generation failed: {e}")
        return {"final_report": f"Fatal Error: {str(e)}", "messages": [AIMessage(content="Failed")]}

async def meta_learning_node(state: AgentState, config: RunnableConfig):
    try:
        conf = state.get("confidence_score", 0.8)
        iters = state.get("research_iterations", 0)
        if conf > 0.85 and iters < 4:
            return Command(goto=END, update={"lessons_learned": state.get("lessons_learned", [])})
        configurable = Configuration.from_runnable_config(config)
        mc = {"model": configurable.research_model, "max_tokens": 500, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        res = await safe_llm_invoke(init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]), [HumanMessage(content=meta_learning_prompt.format(confidence_score=conf, iterations=iters))])
        new_l = state.get("lessons_learned", [])
        if "LESSON:" in res.content and "Strategy optimal" not in res.content:
            new_l.append(res.content.strip())
        if "github.com" in str(state.get("notes", [])):
            omega_memory.update_domain("github.com", True)
        return Command(goto=END, update={"lessons_learned": new_l})
    except Exception:
        return Command(goto=END)

builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
builder.add_node("clarify_with_user", clarify_with_user)
builder.add_node("write_research_brief", write_research_brief)
builder.add_node("meta_cognitive_router", meta_cognitive_router)
builder.add_node("research_supervisor", supervisor_subgraph)
builder.add_node("reasoning_council", reasoning_council)
builder.add_node("adversarial_verification", adversarial_verification)
builder.add_node("final_report_generation", final_report_generation)
builder.add_node("meta_learning_node", meta_learning_node)

builder.add_edge(START, "clarify_with_user")
builder.add_edge("clarify_with_user", "write_research_brief")
builder.add_edge("write_research_brief", "meta_cognitive_router")
builder.add_edge("meta_cognitive_router", "research_supervisor")
builder.add_edge("research_supervisor", "reasoning_council")
builder.add_edge("reasoning_council", "adversarial_verification")
builder.add_edge("adversarial_verification", "final_report_generation")
builder.add_edge("final_report_generation", "meta_learning_node")
builder.add_edge("meta_learning_node", END)

memory = MemorySaver()
deep_researcher = builder.compile(checkpointer=memory)

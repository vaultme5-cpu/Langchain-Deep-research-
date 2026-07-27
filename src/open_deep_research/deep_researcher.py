"""Main LangGraph implementation for Project Omega V2."""
import asyncio
import hashlib
import logging
import re

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

async def safe_llm_invoke(model, messages):
    async with groq_burst_semaphore:
        for attempt in range(3):
            try:
                return await model.ainvoke(messages)
            except Exception as e:
                err = str(e).lower()
                if "rate limit" in err or "429" in err or "resource_exhausted" in err or "timeout" in err:
                    new_key = _shield.get_key()
                    model = model.with_config({"api_key": new_key})
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if attempt == 2:
                    raise RuntimeError("[EPISTEMIC FLAG]: LLM infrastructure constraint.")
        raise RuntimeError("[EPISTEMIC FLAG]: LLM infrastructure constraint.")

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
        plan_dicts = [n.model_dump() for n in response.research_plan] if response.research_plan else []
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
        mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).bind_tools([ConductResearch, ResearchComplete, think_tool])
        saturation = calculate_epistemic_saturation(state.get("evidence_graph", []), state.get("research_plan", []))
        if saturation >= 0.85:
            return Command(goto="supervisor_tools", update={"supervisor_messages": [AIMessage(content="", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "forced_halt"}])], "research_iterations": state.get("research_iterations", 0) + 1})
        
        sup_msgs = list(state.get("supervisor_messages", []))
        core_msgs = [m for m in sup_msgs if isinstance(m, (SystemMessage, HumanMessage)) and "DAG_STATUS" not in str(getattr(m, "content", ""))]
        recent_msgs = [m for m in sup_msgs if not isinstance(m, (SystemMessage, HumanMessage))][-4:]
        sup_msgs = core_msgs + recent_msgs
        
        plan = state.get("research_plan", [])
        completed = state.get("completed_nodes", [])
        if plan:
            dag_status = f"\n<DAG_STATUS>\nPlan: {plan}\nCompleted Nodes: {completed}\n</DAG_STATUS>"
            sup_msgs.append(SystemMessage(content=dag_status))
            
        response = await safe_llm_invoke(rm, sup_msgs)
        return Command(goto="supervisor_tools", update={"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1})
    except Exception as e:
        logging.error(f"supervisor failed: {e}")
        return Command(goto=END)

async def supervisor_tools(state: SupervisorState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        sup_msgs = state.get("supervisor_messages", [])
        if not sup_msgs:
            return Command(goto=END)
        iters = state.get("research_iterations", 0)
        last_msg = sup_msgs[-1]
        if iters > configurable.max_researcher_iterations or not getattr(last_msg, "tool_calls", None) or any(tc["name"] == "ResearchComplete" for tc in last_msg.tool_calls):
            return Command(goto=END, update={"notes": get_notes_from_tool_calls(sup_msgs), "research_brief": state.get("research_brief", "")})
            
        all_tool_msgs = []
        update_payload = {"supervisor_messages": []}
        conduct_calls = [t for t in last_msg.tool_calls if t["name"] == "ConductResearch"]
        if conduct_calls:
            allowed = conduct_calls[:configurable.max_concurrent_research_units]
            tasks = []
            for tc in allowed:
                base_topic = tc["args"]["research_topic"]
                invariant_payload = f"{base_topic}\n\n[INVARIANT CONSTRAINTS]\nTemporal: {state.get('temporal_intent')}\nHard Constraints: {state.get('hard_constraints')}"
                tasks.append(researcher_subgraph.ainvoke({"researcher_messages": [HumanMessage(content=invariant_payload)], "research_topic": invariant_payload}, config))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            vfs_update, agg_graph = {}, []
            for obs, tc in zip(results, allowed):
                if isinstance(obs, Exception):
                    all_tool_msgs.append(ToolMessage(content=f"[FALLBACK] Researcher failed: {str(obs)}", name=tc["name"], tool_call_id=tc["id"]))
                    continue
                art_id = str(obs.get("artifact_id", tc["id"]))
                vfs_update[art_id] = obs.get("compressed_research", "")
                all_tool_msgs.append(ToolMessage(content=f"ARTIFACT ID: {art_id}\nSUMMARY: {obs.get('executive_summary', 'Done')}\n[VFS STORED]", name=tc["name"], tool_call_id=tc["id"]))
                agg_graph.extend(obs.get("evidence_graph", []))
            if vfs_update:
                update_payload["virtual_filesystem"] = vfs_update
            if agg_graph:
                update_payload["evidence_graph"] = agg_graph
            newly_completed = [tc["args"].get("node_id", "") for tc in allowed if tc["args"].get("node_id")]
            if newly_completed:
                update_payload["completed_nodes"] = list(set(state.get("completed_nodes", [])).union(set(newly_completed)))
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


def _chunk_messages_for_compression(messages, char_budget=7000, max_messages=8):
    chunks = []
    current = []
    current_chars = 0

    for msg in messages:
        text = _message_text(msg).strip()
        if not text:
            continue

        if current and (current_chars + len(text) > char_budget or len(current) >= max_messages):
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(msg)
        current_chars += len(text)

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


def _message_text(msg):
    return str(getattr(msg, "content", "") or "")


def _chunk_messages_for_compression(messages, char_budget=7000, max_messages=8):
    chunks = []
    current = []
    current_chars = 0

    for msg in messages:
        text = _message_text(msg).strip()
        if not text:
            continue

        if current and (current_chars + len(text) > char_budget or len(current) >= max_messages):
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(msg)
        current_chars += len(text)

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


async def _compress_message_chunk(sm, chunk_messages, configurable, depth=0):
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
        if (
            is_token_limit_exceeded(e, configurable.compression_model)
            and len(chunk_messages) > 1
            and depth < 4
        ):
            mid = max(1, len(chunk_messages) // 2)
            left_nodes, left_notes = await _compress_message_chunk(
                sm, chunk_messages[:mid], configurable, depth + 1
            )
            right_nodes, right_notes = await _compress_message_chunk(
                sm, chunk_messages[mid:], configurable, depth + 1
            )
            combined_notes = "\n\n".join([n for n in [left_notes, right_notes] if n])
            return left_nodes + right_nodes, combined_notes
        raise


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
            char_budget=7000,
            max_messages=8,
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
    confidence_score = float(state.get("confidence_score", 0.0) or 0.0)
    contradiction_count = sum(1 for n in state.get("evidence_graph", []) if getattr(n, "contradicts", []))
    if confidence_score < 0.65 or contradiction_count > 0:
        if state.get("research_iterations", 0) < 3:
            return Command(
                goto="research_supervisor",
                update={
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

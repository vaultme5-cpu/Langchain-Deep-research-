"""Omega Supremacy Orchestration Engine (Path B: Groq-only)."""
import asyncio, hashlib, logging, re, json
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages, get_buffer_string
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
    compute_epistemic_links,
    AgentInputState, AgentState, ClarifyWithUser, ConductResearch,
    EvidenceGraphExtraction, EvidenceNode, ResearchComplete, ResearcherOutputState,
    ResearcherState, ResearchQuestion, RouterDecision, SupervisorState,
)
from open_deep_research.utils import (
    check_information_satiation, filter_and_verify_evidence, get_all_tools,
    get_api_key_for_model, get_model_token_limit, get_notes_from_tool_calls,
    get_today_str, is_token_limit_exceeded, validate_urls, think_tool,
    omega_local_memory, verify_citations_programmatically,
    calculate_epistemic_saturation, programmatic_epistemic_verification,
    _shield, omega_memory,
)

NL = chr(10)
_BURST_SEMAPHORES = {}

def _current_loop():
    try: return asyncio.get_running_loop()
    except RuntimeError:
        try: return asyncio.get_event_loop()
        except RuntimeError: return asyncio.new_event_loop()

def _get_groq_burst_semaphore(limit=3):
    key = id(_current_loop())
    if key not in _BURST_SEMAPHORES: _BURST_SEMAPHORES[key] = asyncio.Semaphore(limit)
    return _BURST_SEMAPHORES[key]

def _truncate(msgs, max_chars=5500):
    t_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    o_msgs = [m for m in msgs if not isinstance(m, ToolMessage)]
    if not t_msgs: return o_msgs
    rem = max(1500, int(max_chars) - sum(len(str(m.content)) for m in o_msgs))
    pt = max(300, rem // max(len(t_msgs), 1))
    out = []
    for m in t_msgs:
        c = str(m.content)
        if len(c) > pt:
            h = pt // 2
            c = c[:h] + NL + "[TRUNCATED]" + NL + c[-h:]
        out.append(ToolMessage(content=c, tool_call_id=getattr(m, "tool_call_id", "tool"), name=getattr(m, "name", "tool")))
    return o_msgs + out

def _chunk_text(text, chunk_chars=2600):
    text = str(text or "")
    if len(text) <= chunk_chars: return [text]
    out = []
    step = max(1000, int(chunk_chars))
    for i in range(0, len(text), step):
        out.append(text[i:i + step])
    return out[:5]

def _chunk_messages_for_compression(msgs, max_chunk_chars=2600, max_messages=30):
    out = []
    for m in msgs:
        if isinstance(m, ToolMessage) and len(str(m.content)) > max_chunk_chars:
            chunks = _chunk_text(str(m.content), max_chunk_chars)[:3]
            for idx, part in enumerate(chunks, start=1):
                out.append(ToolMessage(content="[CHUNK " + str(idx) + "]" + NL + part, tool_call_id=getattr(m, "tool_call_id", "chunk"), name=getattr(m, "name", "tool")))
        else:
            out.append(m)
        if len(out) >= max_messages: break
    return out

async def safe_llm_invoke(model, messages, max_attempts=4):
    sem = _get_groq_burst_semaphore(1)
    async with sem:
        last_error = None
        for attempt in range(max_attempts):
            try:
                return await model.ainvoke(messages)
            except Exception as e:
                last_error = e
                err = str(e).lower()
                if "rate limit" in err or "429" in err or "resource_exhausted" in err:
                    new_key = _shield.get_key()
                    try: model = model.with_config({"api_key": new_key})
                    except Exception: pass
                    await asyncio.sleep(12.0 * (attempt + 1))
                    continue
                if "413" in err or "timeout" in err:
                    await asyncio.sleep(6.0 * (attempt + 1))
                    continue
                if "400" in err or "context" in err or "too long" in err or "tool_use_failed" in err:
                    if len(messages) > 4:
                        messages = _truncate(messages[:2] + messages[-2:])
                    await asyncio.sleep(2.0)
                    continue
                if attempt == max_attempts - 1:
                    logging.error("safe_llm_invoke fatal: " + str(e))
                    raise RuntimeError("[EPISTEMIC FLAG]: " + str(e))
        logging.error("safe_llm_invoke exhausted: " + str(last_error))
        raise RuntimeError("[EPISTEMIC FLAG]: Max retries exhausted. " + str(last_error))

def _gemini_ladder(cfg):
    raw = str(getattr(cfg, "gemini_models", "") or "").strip()
    if not raw: raw = str(getattr(cfg, "gemini_model", "google_genai:gemini-2.0-flash"))
    return [(m.strip(), cfg.gemini_model_max_tokens) for m in raw.split(",") if m.strip()]

def _brain_chain(cfg, kind):
    g = _gemini_ladder(cfg)
    if kind == "compress": return g + [(cfg.compression_model, cfg.compression_model_max_tokens)]
    if kind == "intake": return [(cfg.intake_model, cfg.intake_model_max_tokens)] + g
    if kind == "report": return [(cfg.final_report_model, cfg.final_report_model_max_tokens)] + g
    return [(cfg.research_model, cfg.research_model_max_tokens)] + g

async def _brain_invoke(cfg, config, kind, messages, structured=None, tools=None):
    chain = _brain_chain(cfg, kind)
    last = None
    for name, tok in chain:
        m = init_chat_model(model=name, max_tokens=tok, api_key=get_api_key_for_model(name, config))
        if structured is not None: m = m.with_structured_output(structured)
        if tools is not None: m = m.bind_tools(tools)
        try:
            return await safe_llm_invoke(m, messages)
        except Exception as e:
            last = e
            err = str(e).lower()
            if "429" in err or "rate limit" in err or "resource_exhausted" in err or "404" in err or "not found" in err or "503" in err or "500" in err or "400" in err or "401" in err or "api key" in err or "invalid" in err or "permission" in err:
                logging.error("brain failover from " + name)
                continue
            raise
    raise RuntimeError("[EPISTEMIC FLAG]: All brains exhausted. " + str(last))

def _resurrect_json(raw_text):
    try:
        text = str(raw_text or "")
        text = text.replace("```json", "").replace("```", "")
        text = re.sub("<function.*?>", "", text, flags=re.DOTALL)
        text = re.sub("</function.*?>", "", text, flags=re.DOTALL)
        start = -1
        for i, ch in enumerate(text):
            if ch == "{" or ch == "[":
                start = i
                break
        if start == -1: return None
        text = text[start:]
        if text.startswith("["):
            text = "{" + chr(34) + "nodes" + chr(34) + ": " + text + "}"
        text = re.sub(",[ ]*}", "}", text)
        text = re.sub(",[ ]*]", "]", text)
        return EvidenceGraphExtraction.model_validate_json(text)
    except Exception:
        return None

def _erc_build_snapshot(state):
    ev = state.get("evidence_graph", []) or []
    plan = state.get("research_plan", []) or []
    comp = state.get("completed_nodes", []) or []
    ev_repr = " | ".join(sorted([str(getattr(n, "claim", "")) for n in ev if getattr(n, "claim", "")]))
    plan_repr = json.dumps(plan, sort_keys=True, default=str)
    comp_repr = " | ".join(sorted([str(x) for x in comp]))
    payload = ev_repr + " || " + plan_repr + " || " + comp_repr
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def generate_argus_view(nodes):
    if not nodes: return "No structured evidence."
    sc = {getattr(n, "citation_index", 0): 0 for n in nodes}
    for n in nodes:
        for s in getattr(n, "supports", []):
            if s in sc: sc[s] += 1
    foundational = [n for n in nodes if sc.get(getattr(n, "citation_index", 0), 0) >= 2]
    contradicted = [n for n in nodes if getattr(n, "contradicts", [])]
    view = "### ARGUS VIEW" + NL
    for n in foundational[:5]:
        view += "- [" + str(getattr(n, "citation_index", 0)) + "] " + str(getattr(n, "claim", "")) + " (x" + str(sc.get(getattr(n, "citation_index", 0), 0)) + ")" + NL
    for n in contradicted[:3]:
        view += "- [" + str(getattr(n, "citation_index", 0)) + "] " + str(getattr(n, "claim", "")) + " (CONTRADICTS)" + NL
    return view

def add_targeted_research_nodes(evidence_graph, research_plan):
    plan = list(research_plan) if research_plan else []
    existing = set()
    for n in plan:
        if isinstance(n, dict) and n.get("node_id"): existing.add(str(n.get("node_id")))
    targets = []
    for idx, node in enumerate((evidence_graph or [])[:3], start=1):
        claim = str(getattr(node, "claim", "")).strip()
        if not claim: continue
        targets.append({"node_id": "FB_" + str(idx), "topic": "Verify and resolve: " + claim[:150], "depends_on": []})
    if not targets: targets = [{"node_id": "FB_1", "topic": "Resolve contradictions and diversify sources.", "depends_on": []}]
    for t in targets:
        if str(t.get("node_id")) not in existing: plan.append(t)
    return plan

async def clarify_with_user(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        if not cfg.allow_clarification: return Command(goto="write_research_brief")
        pass
        prompt = clarify_with_user_instructions.format(messages=get_buffer_string(state.get("messages", [])), date=get_today_str())
        r = await _brain_invoke(cfg, config, "intake", [HumanMessage(content=prompt)], structured=ClarifyWithUser)
        if getattr(r, "need_clarification", False):
            return Command(goto=END, update={"messages": [AIMessage(content=getattr(r, "question", "Please clarify."))]})
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=getattr(r, "verification", "Proceeding."))]})
    except Exception as e:
        logging.error("clarify failed: " + str(e))
        return Command(goto="write_research_brief")

async def write_research_brief(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        pass
        prompt = transform_messages_into_research_topic_prompt.format(messages=get_buffer_string(state.get("messages", [])), date=get_today_str())
        r = await _brain_invoke(cfg, config, "intake", [HumanMessage(content=prompt)], structured=ResearchQuestion)
        mem = omega_memory.get_context_prompt()
        sup_sys = lead_researcher_prompt.format(
            date=get_today_str(),
            mcp_prompt=cfg.mcp_prompt or " ",
            max_concurrent_research_units=cfg.max_concurrent_research_units,
            max_researcher_iterations=cfg.max_researcher_iterations,
            temporal_intent=getattr(r, "temporal_intent", "Current"),
            complexity_tier="Pending",
            lessons_learned=mem,
            hard_constraints=getattr(r, "hard_constraints", []),
            memory_context=mem
        )
        return Command(goto="meta_cognitive_router", update={"research_brief": getattr(r, "research_brief", ""), "temporal_intent": getattr(r, "temporal_intent", "Current"), "hard_constraints": getattr(r, "hard_constraints", []), "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=getattr(r, "research_brief", ""))]}})
    except Exception as e:
        logging.error("brief failed: " + str(e))
        return Command(goto=END)

async def meta_cognitive_router(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        pass
        mem = omega_memory.get_context_prompt()
        prompt = meta_cognitive_router_prompt.format(research_brief=state.get("research_brief", " "), date=get_today_str(), memory_context=mem)
        r = await _brain_invoke(cfg, config, "intake", [HumanMessage(content=prompt)], structured=RouterDecision)
        sup_sys = lead_researcher_prompt.format(
            date=get_today_str(),
            mcp_prompt=cfg.mcp_prompt or " ",
            max_concurrent_research_units=getattr(r, "dynamic_research_units", cfg.max_concurrent_research_units),
            max_researcher_iterations=getattr(r, "dynamic_tool_budget", cfg.max_researcher_iterations),
            complexity_tier=getattr(r, "complexity_tier", "Medium"),
            temporal_intent=state.get("temporal_intent", "Current"),
            lessons_learned=mem,
            hard_constraints=state.get("hard_constraints", []),
            memory_context=mem
        )
        pd = []
        for n in (getattr(r, "research_plan", []) or []):
            try: pd.append(n.model_dump())
            except Exception: pd.append({"node_id": str(getattr(n, "node_id", "")), "topic": str(getattr(n, "topic", "")), "depends_on": list(getattr(n, "depends_on", []))})
        return Command(goto="research_supervisor", update={"query_paradigm": getattr(r, "query_paradigm", "General"), "complexity_tier": getattr(r, "complexity_tier", "Medium"), "dynamic_tool_budget": getattr(r, "dynamic_tool_budget", cfg.max_react_tool_calls), "dynamic_research_units": getattr(r, "dynamic_research_units", cfg.max_concurrent_research_units), "research_plan": pd, "completed_nodes": [], "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=state.get("research_brief", " "))]}})
    except Exception as e:
        logging.error("router failed: " + str(e))
        return Command(goto=END)

async def researcher(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        tools = await get_all_tools(config)
        pass
        mem = omega_memory.get_context_prompt()
        prompt = research_system_prompt.format(mcp_prompt=cfg.mcp_prompt or " ", date=get_today_str(), temporal_intent=state.get("temporal_intent", "Current"), hard_constraints=state.get("hard_constraints", []), memory_context=mem)
        pass
        r_msgs = state.get("researcher_messages", [])
        core = [m for m in r_msgs if isinstance(m, (SystemMessage, HumanMessage))]
        recent = [m for m in r_msgs if not isinstance(m, (SystemMessage, HumanMessage))][-6:]
        msgs = [SystemMessage(content=prompt)] + core + recent
        msgs = _truncate(msgs, getattr(cfg, "max_tool_payload_chars", 5500))
        r = await _brain_invoke(cfg, config, "work", msgs, tools=tools)
        return Command(goto="researcher_tools", update={"researcher_messages": [r], "tool_call_iterations": int(state.get("tool_call_iterations", 0) or 0) + 1})
    except Exception as e:
        logging.error("researcher failed: " + str(e))
        return Command(goto="compress_research")

async def researcher_tools(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        r_msgs = state.get("researcher_messages", [])
        if not r_msgs: return Command(goto="compress_research")
        last = r_msgs[-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls: return Command(goto="compress_research")
        tools = await get_all_tools(config)
        tbn = {t.name: t for t in tools if hasattr(t, "name")}
        obs = []
        for t in calls:
            name = str(t.get("name", ""))
            args = t.get("args", {}) or {}
            if name in tbn:
                try: obs.append(await tbn[name].ainvoke(args, config))
                except Exception: obs.append("[FALLBACK] Tool failed.")
            else: obs.append("[FALLBACK] Tool missing.")
        to = []
        for o, t in zip(obs, calls):
            to.append(ToolMessage(content=str(o), name=str(t.get("name", "tool")), tool_call_id=str(t.get("id", "tool"))))
        nc = [str(o) for o in obs if isinstance(o, str)]
        ec = [str(getattr(m, "content", "")) for m in r_msgs if isinstance(getattr(m, "content", ""), str)]
        if check_information_satiation(nc, ec) or int(state.get("tool_call_iterations", 0) or 0) >= cfg.max_react_tool_calls:
            return Command(goto="compress_research", update={"researcher_messages": to})
        return Command(goto="researcher", update={"researcher_messages": to})
    except Exception as e:
        logging.error("researcher_tools failed: " + str(e))
        return Command(goto="compress_research")

async def compress_research(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        pass
        r_msgs = state.get("researcher_messages", [])
        sys_msg = SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))
        tool_msgs = [m for m in r_msgs if isinstance(m, ToolMessage)]
        tool_text = NL.join([str(m.content) for m in tool_msgs])
        chunk_limit = int(getattr(cfg, "max_compression_chunk_chars", 2600))
        all_nodes = []
        if len(tool_text) > chunk_limit * 2:
            chunks_all = _chunk_text(tool_text, chunk_limit)
            if len(chunks_all) > 3:
                chunks = [chunks_all[0], chunks_all[len(chunks_all) // 2], chunks_all[-1]]
            else:
                chunks = chunks_all
            for c in chunks:
                msgs = [sys_msg, HumanMessage(content="Extract facts from this research chunk." + NL + c), HumanMessage(content=compress_research_simple_human_message)]
                try:
                    r = await _brain_invoke(cfg, config, "compress", msgs, structured=EvidenceGraphExtraction)
                    if getattr(r, "nodes", None): all_nodes.extend(r.nodes)
                except Exception as ce:
                    rescued = _resurrect_json(str(ce))
                    if rescued and getattr(rescued, "nodes", None): all_nodes.extend(rescued.nodes)
        if not all_nodes:
            msgs = [sys_msg] + _chunk_messages_for_compression(r_msgs, chunk_limit) + [HumanMessage(content=compress_research_simple_human_message)]
            msgs = _truncate(msgs, getattr(cfg, "max_tool_payload_chars", 5500))
            try:
                r = await _brain_invoke(cfg, config, "compress", msgs, structured=EvidenceGraphExtraction)
                all_nodes = getattr(r, "nodes", []) or []
            except Exception as e:
                rescued = _resurrect_json(str(e))
                if rescued and getattr(rescued, "nodes", None): all_nodes = rescued.nodes
                else: raise e
        r_nodes = compute_epistemic_links(all_nodes)
        aid = hashlib.sha256((tool_text or "none").encode("utf-8")).hexdigest()[:10]
        if r_nodes:
            rd = "Evidence:" + NL + NL.join(["Fact " + str(i+1) + ": " + str(getattr(n, "claim", "")) + " (" + str(getattr(n, "url", "")) + ")" for i, n in enumerate(r_nodes)])
        else:
            rd = "No evidence extracted."
        for n in r_nodes:
            omega_local_memory.store(getattr(n, "claim", ""), getattr(n, "url", ""))
        return {"compressed_research": rd[:12000], "artifact_id": aid, "executive_summary": rd[:500], "evidence_graph": r_nodes}
    except Exception as e:
        logging.error("compress failed: " + str(e))
        return {"compressed_research": "Error", "artifact_id": "err", "executive_summary": "Failed", "evidence_graph": []}

rb = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
rb.add_node("researcher", researcher)
rb.add_node("researcher_tools", researcher_tools)
rb.add_node("compress_research", compress_research)
rb.add_edge(START, "researcher")
rb.add_edge("compress_research", END)
researcher_subgraph = rb.compile()

async def supervisor(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        iters = int(state.get("research_iterations", 0) or 0)
        snapshot = _erc_build_snapshot(state)
        prev = str(state.get("erc_frontier_fingerprint", "") or "")
        no_progress = int(state.get("erc_no_progress_count", 0) or 0)
        if prev == snapshot: no_progress += 1
        else: no_progress = 0
        erc_update = {"erc_frontier_fingerprint": snapshot, "erc_no_progress_count": no_progress, "research_iterations": iters + 1}
        sat = calculate_epistemic_saturation(state.get("evidence_graph", []), state.get("research_plan", []))
        if sat >= 0.85 or iters >= cfg.max_researcher_iterations:
            erc_update["supervisor_messages"] = [AIMessage(content="", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "halt"}])]
            return Command(goto="supervisor_tools", update=erc_update)
        if no_progress >= cfg.erc_max_stagnation_iterations:
            if iters + 1 < cfg.max_researcher_iterations:
                plan = add_targeted_research_nodes(state.get("evidence_graph", []), state.get("research_plan", []))
                topic = "Diversify search and resolve stagnant evidence. Focus on contradictions, missing dates, and source diversity."
                erc_update["research_plan"] = plan
                erc_update["supervisor_messages"] = [AIMessage(content="", tool_calls=[{"name": "ConductResearch", "args": {"node_id": "FB_ERC", "research_topic": topic}, "id": "erc"}])]
                return Command(goto="supervisor_tools", update=erc_update)
            erc_update["supervisor_messages"] = [AIMessage(content="", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "halt"}])]
            return Command(goto="supervisor_tools", update=erc_update)
        pass
        sup_msgs = list(state.get("supervisor_messages", []))
        core = [m for m in sup_msgs if isinstance(m, (SystemMessage, HumanMessage)) and "DAG_STATUS" not in str(getattr(m, "content", ""))]
        recent = [m for m in sup_msgs if not isinstance(m, (SystemMessage, HumanMessage))][-4:]
        sup_msgs = core + recent
        plan = state.get("research_plan", [])
        comp = state.get("completed_nodes", [])
        if plan:
            sup_msgs.append(SystemMessage(content="<DAG_STATUS>" + NL + "Plan: " + json.dumps(plan, default=str)[:3000] + NL + "Completed: " + str(comp)[:1000] + NL + "</DAG_STATUS>"))
        r = await _brain_invoke(cfg, config, "work", sup_msgs, tools=[ConductResearch, ResearchComplete, think_tool])
        erc_update["supervisor_messages"] = [r]
        return Command(goto="supervisor_tools", update=erc_update)
    except Exception as e:
        logging.error("supervisor failed: " + str(e))
        return Command(goto=END)

async def supervisor_tools(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        sup_msgs = state.get("supervisor_messages", [])
        if not sup_msgs: return Command(goto=END)
        iters = int(state.get("research_iterations", 0) or 0)
        last = sup_msgs[-1]
        calls = getattr(last, "tool_calls", None) or []
        if iters > cfg.max_researcher_iterations or not calls or any(str(t.get("name", "")) == "ResearchComplete" for t in calls):
            return Command(goto=END, update={"notes": get_notes_from_tool_calls(sup_msgs), "research_brief": state.get("research_brief", "")})
        cc = [t for t in calls if str(t.get("name", "")) == "ConductResearch"]
        if not cc:
            tm = []
            for t in calls:
                tm.append(ToolMessage(content="Acknowledged.", name=str(t.get("name", "tool")), tool_call_id=str(t.get("id", "tool"))))
            return Command(goto="supervisor", update={"supervisor_messages": tm or [AIMessage(content="Continuing.")]})
        allowed = cc[:cfg.max_concurrent_research_units]
        tasks = []
        for t in allowed:
            args = t.get("args", {}) or {}
            bt = str(args.get("research_topic", ""))
            inv = bt + NL + NL + "[INVARIANT]" + NL + "Temporal: " + str(state.get("temporal_intent", "Current")) + NL + "Constraints: " + str(state.get("hard_constraints", []))
            tasks.append(researcher_subgraph.ainvoke({"researcher_messages": [HumanMessage(content=inv)], "research_topic": inv}, config))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        atm, up, vu, ag = [], {"supervisor_messages": []}, {}, []
        for obs, t in zip(results, allowed):
            if isinstance(obs, Exception):
                atm.append(ToolMessage(content="[FALLBACK] " + str(obs), name=str(t.get("name", "ConductResearch")), tool_call_id=str(t.get("id", "tool"))))
                continue
            aid = str(obs.get("artifact_id", str(t.get("id", "art"))))
            vu[aid] = str(obs.get("compressed_research", ""))
            atm.append(ToolMessage(content="ARTIFACT: " + aid + NL + str(obs.get("executive_summary", "Done")), name=str(t.get("name", "ConductResearch")), tool_call_id=str(t.get("id", "tool"))))
            ag.extend(obs.get("evidence_graph", []))
        if vu: up["virtual_filesystem"] = vu
        if ag: up["evidence_graph"] = ag
        nc = []
        for t in allowed:
            nid = (t.get("args", {}) or {}).get("node_id")
            if nid: nc.append(str(nid))
        if nc: up["completed_nodes"] = list(set(list(state.get("completed_nodes", [])) + nc))
        up["supervisor_messages"] = atm
        return Command(goto="supervisor", update=up)
    except Exception as e:
        logging.error("supervisor_tools failed: " + str(e))
        return Command(goto=END)

sb = StateGraph(SupervisorState, config_schema=Configuration)
sb.add_node("supervisor", supervisor)
sb.add_node("supervisor_tools", supervisor_tools)
sb.add_edge(START, "supervisor")
supervisor_subgraph = sb.compile()

async def reasoning_council(state, config):
    try:
        tier = str(state.get("complexity_tier", "Medium"))
        if tier in ["Simple", "Medium"]:
            return Command(goto="adversarial_verification", update={"master_synthesis": "Standard inductive synthesis."})
        cfg = Configuration.from_runnable_config(config)
        argus = generate_argus_view(state.get("evidence_graph", []))
        findings = argus + NL + NL.join([str(x) for x in state.get("notes", [])])[:6000]
        brief = state.get("research_brief", " ")
        async def run_p(p):
            pass
            try:
                prompt = reasoning_council_prompt.format(paradigm=p, brief=brief, findings=findings[:10000])
                res = await _brain_invoke(cfg, config, "intake", [HumanMessage(content=prompt)])
                return "### " + p + NL + str(getattr(res, "content", ""))
            except Exception:
                return "### " + p + NL + "Skipped."
        results = await asyncio.gather(*[run_p(p) for p in ["Deductive", "Inductive", "Abductive"]])
        return Command(goto="adversarial_verification", update={"master_synthesis": (NL + NL).join(results)})
    except Exception as e:
        logging.error("council failed: " + str(e))
        return Command(goto="adversarial_verification", update={"master_synthesis": "Council failed."})

async def adversarial_verification(state, config):
    try:
        ev = state.get("evidence_graph", [])
        ti = state.get("temporal_intent", "Current")
        vr = programmatic_epistemic_verification(ev, ti)
        argus = generate_argus_view(ev) if ev else " "
        return Command(goto="final_report_generation", update={"red_team_findings": vr.get("red_team_findings", ""), "devils_advocate_critique": vr.get("devils_advocate_critique", ""), "consensus_report": str(vr.get("consensus_report", "")) + NL + argus, "confidence_score": float(vr.get("confidence_score", 0.5))})
    except Exception as e:
        logging.error("verify failed: " + str(e))
        return Command(goto="final_report_generation", update={"confidence_score": 0.5})

async def final_report_generation(state, config):
    try:
        conf = float(state.get("confidence_score", 0.0) or 0.0)
        contradictions = sum(1 for n in state.get("evidence_graph", []) if getattr(n, "contradicts", []))
        cfg = Configuration.from_runnable_config(config)
        iters = int(state.get("research_iterations", 0) or 0)
        if (conf < 0.65 or contradictions > 0) and iters < cfg.max_researcher_iterations:
            return Command(goto="research_supervisor", update={"research_plan": add_targeted_research_nodes(state.get("evidence_graph", []), state.get("research_plan", [])), "complexity_tier": "Complex"})
        ev = state.get("evidence_graph", [])
        verified = []
        if ev:
            urls = [str(getattr(n, "url", "")) for n in ev if getattr(n, "url", "")]
            if urls:
                try:
                    h = await validate_urls(urls)
                    ev = [n for n in ev if h.get(str(getattr(n, "url", "")), False)]
                except Exception: pass
            verified = filter_and_verify_evidence(ev, temporal_intent=state.get("temporal_intent", "Current"))
            try:
                checked = await verify_citations_programmatically(verified)
                if checked: verified = checked
            except Exception: pass
        vn = []
        for n in verified:
            d = str(getattr(n, "date_published", "")) if getattr(n, "date_published", None) else "Unknown"
            vn.append("Fact: " + str(getattr(n, "claim", "")) + NL + "Source: " + str(getattr(n, "url", "")) + NL + "Date: " + d)
        notes = [str(x) for x in state.get("notes", [])] + vn
        vfs = state.get("virtual_filesystem", {})
        ve = (NL + NL).join(["### VFS " + str(k) + NL + str(v) for k, v in vfs.items()])
        findings = NL.join(notes) + NL + NL + ve
        wc = {"model": cfg.final_report_model, "max_tokens": cfg.final_report_model_max_tokens, "api_key": get_api_key_for_model(cfg.final_report_model, config)}
        prompt = final_report_generation_prompt.format(
            research_brief=state.get("research_brief", " "),
            findings=findings[:12000],
            date=get_today_str(),
            master_synthesis=state.get("master_synthesis", " "),
            consensus_report=state.get("consensus_report", "None"),
            confidence_score=state.get("confidence_score", 0.8),
            query_paradigm=state.get("query_paradigm", "General")
        )
        rep = await _brain_invoke(cfg, config, "report", [HumanMessage(content=prompt)])
        return {"final_report": getattr(rep, "content", ""), "messages": [rep], "notes": {"type": "override", "value": []}}
    except Exception as e:
        logging.error("report failed: " + str(e))
        return {"final_report": "Fatal: " + str(e), "messages": [AIMessage(content="Failed")]}

async def meta_learning_node(state, config):
    try:
        conf = float(state.get("confidence_score", 0.8) or 0.8)
        iters = int(state.get("research_iterations", 0) or 0)
        if conf > 0.85 and iters < 4: return Command(goto=END)
        cfg = Configuration.from_runnable_config(config)
        pass
        res = await _brain_invoke(cfg, config, "intake", [HumanMessage(content=meta_learning_prompt.format(confidence_score=conf, iterations=iters))])
        nl = list(state.get("lessons_learned", []))
        content = str(getattr(res, "content", ""))
        if "LESSON:" in content: nl.append(content.strip())
        return Command(goto=END, update={"lessons_learned": nl})
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

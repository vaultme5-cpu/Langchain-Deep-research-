"""Omega Supremacy Orchestration Engine (God-Tier AEGIS Hardened)."""
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

configurable_model = init_chat_model(configurable_fields=("model", "max_tokens", "api_key"))

# === LOOP-SAFE SEMAPHORE (Kills the 15-min Event Loop Crash) ===
_SEMAPHORES = {}
def _get_sem():
    try: loop = asyncio.get_running_loop()
    except RuntimeError: loop = asyncio.new_event_loop()
    key = id(loop)
    if key not in _SEMAPHORES: _SEMAPHORES[key] = asyncio.Semaphore(3)
    return _SEMAPHORES[key]

# === CONTEXT DISTILLER (Kills the 413 TPM Wall) ===
def _truncate(msgs, max_chars=5500):
    t_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    o_msgs = [m for m in msgs if not isinstance(m, ToolMessage)]
    rem = max(1500, max_chars - sum(len(str(m.content)) for m in o_msgs))
    pt = max(300, rem // max(len(t_msgs), 1))
    out = []
    for m in t_msgs:
        c = str(m.content)
        if len(c) > pt:
            h = pt // 2
            c = c[:h] + "\n[TRUNCATED]\n" + c[-h:]
        out.append(ToolMessage(content=c, tool_call_id=m.tool_call_id, name=getattr(m, "name", "tool")))
    return o_msgs + out

# === JSON RESURRECTION (Kills the 400 tool_use_failed Crash) ===
def _resurrect_json(raw_text):
    try:
        text = re.sub(r"<function=.*?>", "", raw_text)
        text = re.sub(r"</function>", "", text)
        start = -1
        for i, ch in enumerate(text):
            if ch in "{[": start = i; break
        if start == -1: return None
        text = text[start:]
        if text.startswith("["): text = "{" + text[1:]
        if text.endswith("]"): text = text[:-1] + "}"
        text = re.sub(r",\s*([\]}])", r"\1", text)
        return EvidenceGraphExtraction.model_validate_json(text)
    except Exception: return None

# === SAFE LLM INVOKE (Catches 413 + 400 + Context Bomb) ===
async def safe_llm_invoke(model, messages):
    sem = _get_sem()
    async with sem:
        for attempt in range(3):
            try:
                return await model.ainvoke(messages)
            except Exception as e:
                err = str(e).lower()
                if "rate limit" in err or "429" in err or "413" in err or "resource_exhausted" in err or "timeout" in err:
                    new_key = _shield.get_key()
                    model = model.with_config({"api_key": new_key})
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if "400" in err or "context" in err or "too long" in err or "tool_use_failed" in err:
                    if len(messages) > 4:
                        messages = messages[:2] + messages[-2:]
                    await asyncio.sleep(1)
                    continue
                if attempt == 2:
                    logging.error(f"safe_llm_invoke fatal: {e}")
                    raise RuntimeError("[EPISTEMIC FLAG]: " + str(e))
        raise RuntimeError("[EPISTEMIC FLAG]: Max retries exhausted.")

# === ARGUS TOPOLOGICAL VIEW ===
def generate_argus_view(nodes):
    if not nodes: return "No structured evidence."
    sc = {n.citation_index: 0 for n in nodes}
    for n in nodes:
        for s in getattr(n, "supports", []):
            if s in sc: sc[s] += 1
    foundational = [n for n in nodes if sc.get(n.citation_index, 0) >= 2]
    contradicted = [n for n in nodes if getattr(n, "contradicts", [])]
    view = "### ARGUS VIEW\n"
    for n in foundational[:5]: view += f"- [{n.citation_index}] {n.claim} (x{sc[n.citation_index]})\n"
    for n in contradicted[:3]: view += f"- [{n.citation_index}] {n.claim} (CONTRADICTS)\n"
    return view

# === ERC CIRCUIT BREAKER (Kills the Zombie Loop) ===
def add_targeted_research_nodes(evidence_graph, research_plan):
    plan = list(research_plan) if research_plan else []
    existing = {n.get("node_id") for n in plan if isinstance(n, dict)}
    targets = []
    for idx, node in enumerate((evidence_graph or [])[:3], start=1):
        claim = str(getattr(node, "claim", "")).strip()
        if not claim: continue
        targets.append({"node_id": f"FB_{idx}", "topic": f"Verify and resolve: {claim[:150]}", "depends_on": []})
    if not targets: targets = [{"node_id": "FB_1", "topic": "Resolve contradictions.", "depends_on": []}]
    for t in targets:
        if t["node_id"] not in existing: plan.append(t)
    return plan

# === CLARIFY ===
async def clarify_with_user(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        if not cfg.allow_clarification: return Command(goto="write_research_brief")
        mc = {"model": cfg.research_model, "max_tokens": cfg.research_model_max_tokens, "api_key": get_api_key_for_model(cfg.research_model, config)}
        cm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(ClarifyWithUser)
        prompt = clarify_with_user_instructions.format(messages=get_buffer_string(state["messages"]), date=get_today_str())
        r = await safe_llm_invoke(cm, [HumanMessage(content=prompt)])
        if r.need_clarification: return Command(goto=END, update={"messages": [AIMessage(content=r.question)]})
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=r.verification)]})
    except Exception as e:
        logging.error(f"clarify failed: {e}")
        return Command(goto="write_research_brief")

# === WRITE BRIEF ===
async def write_research_brief(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        mc = {"model": cfg.research_model, "max_tokens": cfg.research_model_max_tokens, "api_key": get_api_key_for_model(cfg.research_model, config)}
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(ResearchQuestion)
        prompt = transform_messages_into_research_topic_prompt.format(messages=get_buffer_string(state.get("messages", [])), date=get_today_str())
        r = await safe_llm_invoke(rm, [HumanMessage(content=prompt)])
        mem = omega_memory.get_context_prompt()
        sup_sys = lead_researcher_prompt.format(date=get_today_str(), mcp_prompt=cfg.mcp_prompt or "", max_concurrent_research_units=cfg.max_concurrent_research_units, max_researcher_iterations=cfg.max_researcher_iterations, temporal_intent=getattr(r, "temporal_intent", "Current"), complexity_tier="Pending", lessons_learned=mem, hard_constraints=getattr(r, "hard_constraints", []), memory_context=mem)
        return Command(goto="meta_cognitive_router", update={"research_brief": r.research_brief, "temporal_intent": getattr(r, "temporal_intent", "Current"), "hard_constraints": getattr(r, "hard_constraints", []), "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=r.research_brief)]}})
    except Exception as e:
        logging.error(f"brief failed: {e}")
        return Command(goto=END)

# === META-COGNITIVE ROUTER ===
async def meta_cognitive_router(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        mc = {"model": cfg.research_model, "max_tokens": 4096, "api_key": get_api_key_for_model(cfg.research_model, config)}
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(RouterDecision)
        mem = omega_memory.get_context_prompt()
        prompt = meta_cognitive_router_prompt.format(research_brief=state.get("research_brief", ""), date=get_today_str(), memory_context=mem)
        r = await safe_llm_invoke(rm, [HumanMessage(content=prompt)])
        sup_sys = lead_researcher_prompt.format(date=get_today_str(), mcp_prompt=cfg.mcp_prompt or "", max_concurrent_research_units=r.dynamic_research_units, max_researcher_iterations=r.dynamic_tool_budget, complexity_tier=r.complexity_tier, temporal_intent=state.get("temporal_intent", "Current"), lessons_learned=mem, hard_constraints=state.get("hard_constraints", []), memory_context=mem)
        pd = [n.model_dump() for n in r.research_plan] if r.research_plan else []
        return Command(goto="research_supervisor", update={"query_paradigm": r.query_paradigm, "complexity_tier": r.complexity_tier, "dynamic_tool_budget": r.dynamic_tool_budget, "dynamic_research_units": r.dynamic_research_units, "research_plan": pd, "completed_nodes": [], "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=state.get("research_brief", ""))]}})
    except Exception as e:
        logging.error(f"router failed: {e}")
        return Command(goto=END)

# === SUPERVISOR ===
async def supervisor(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        mc = {"model": cfg.research_model, "max_tokens": cfg.research_model_max_tokens, "api_key": get_api_key_for_model(cfg.research_model, config)}
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).bind_tools([ConductResearch, ResearchComplete, think_tool])
        sat = calculate_epistemic_saturation(state.get("evidence_graph", []), state.get("research_plan", []))
        iters = state.get("research_iterations", 0)
        if sat >= 0.85 or iters >= cfg.max_researcher_iterations:
            return Command(goto="supervisor_tools", update={"supervisor_messages": [AIMessage(content="", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "halt"}])], "research_iterations": iters + 1})
        sup_msgs = list(state.get("supervisor_messages", []))
        core = [m for m in sup_msgs if isinstance(m, (SystemMessage, HumanMessage)) and "DAG_STATUS" not in str(getattr(m, "content", ""))]
        recent = [m for m in sup_msgs if not isinstance(m, (SystemMessage, HumanMessage))][-4:]
        sup_msgs = core + recent
        plan = state.get("research_plan", [])
        comp = state.get("completed_nodes", [])
        if plan: sup_msgs.append(SystemMessage(content="\n<DAG_STATUS>\nPlan: " + str(plan) + "\nCompleted: " + str(comp) + "\n</DAG_STATUS>"))
        r = await safe_llm_invoke(rm, sup_msgs)
        return Command(goto="supervisor_tools", update={"supervisor_messages": [r], "research_iterations": iters + 1})
    except Exception as e:
        logging.error(f"supervisor failed: {e}")
        return Command(goto=END)

# === SUPERVISOR TOOLS ===
async def supervisor_tools(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        sup_msgs = state.get("supervisor_messages", [])
        if not sup_msgs: return Command(goto=END)
        iters = state.get("research_iterations", 0)
        last = sup_msgs[-1]
        if iters > cfg.max_researcher_iterations or not getattr(last, "tool_calls", None) or any(t["name"] == "ResearchComplete" for t in last.tool_calls):
            return Command(goto=END, update={"notes": get_notes_from_tool_calls(sup_msgs), "research_brief": state.get("research_brief", "")})
        atm, up = [], {"supervisor_messages": []}
        cc = [t for t in last.tool_calls if t["name"] == "ConductResearch"]
        if cc:
            allowed = cc[:cfg.max_concurrent_research_units]
            tasks = []
            for t in allowed:
                bt = t["args"]["research_topic"]
                inv = bt + "\n\n[INVARIANT]\nTemporal: " + str(state.get("temporal_intent")) + "\nConstraints: " + str(state.get("hard_constraints"))
                tasks.append(researcher_subgraph.ainvoke({"researcher_messages": [HumanMessage(content=inv)], "research_topic": inv}, config))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            vu, ag = {}, []
            for obs, t in zip(results, allowed):
                if isinstance(obs, Exception):
                    atm.append(ToolMessage(content="[FALLBACK] " + str(obs), name=t["name"], tool_call_id=t["id"]))
                    continue
                aid = str(obs.get("artifact_id", t["id"]))
                vu[aid] = obs.get("compressed_research", "")
                atm.append(ToolMessage(content="ARTIFACT: " + aid + "\n" + str(obs.get("executive_summary", "Done")), name=t["name"], tool_call_id=t["id"]))
                ag.extend(obs.get("evidence_graph", []))
            if vu: up["virtual_filesystem"] = vu
            if ag: up["evidence_graph"] = ag
            nc = [t["args"].get("node_id", "") for t in allowed if t["args"].get("node_id")]
            if nc: up["completed_nodes"] = list(set(state.get("completed_nodes", [])).union(set(nc)))
        up["supervisor_messages"] = atm
        return Command(goto="supervisor", update=up)
    except Exception as e:
        logging.error(f"supervisor_tools failed: {e}")
        return Command(goto=END)

sb = StateGraph(SupervisorState, config_schema=Configuration)
sb.add_node("supervisor", supervisor)
sb.add_node("supervisor_tools", supervisor_tools)
sb.add_edge(START, "supervisor")
supervisor_subgraph = sb.compile()

# === RESEARCHER ===
async def researcher(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        tools = await get_all_tools(config)
        mc = {"model": cfg.research_model, "max_tokens": cfg.research_model_max_tokens, "api_key": get_api_key_for_model(cfg.research_model, config)}
        mem = omega_memory.get_context_prompt()
        prompt = research_system_prompt.format(mcp_prompt=cfg.mcp_prompt or "", date=get_today_str(), temporal_intent=state.get("temporal_intent", "Current"), hard_constraints=state.get("hard_constraints", []), memory_context=mem)
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).bind_tools(tools)
        r_msgs = state.get("researcher_messages", [])
        core = [m for m in r_msgs if isinstance(m, (SystemMessage, HumanMessage))]
        recent = [m for m in r_msgs if not isinstance(m, (SystemMessage, HumanMessage))][-6:]
        msgs = [SystemMessage(content=prompt)] + core + recent
        r = await safe_llm_invoke(rm, msgs)
        return Command(goto="researcher_tools", update={"researcher_messages": [r], "tool_call_iterations": state.get("tool_call_iterations", 0) + 1})
    except Exception as e:
        logging.error(f"researcher failed: {e}")
        return Command(goto="compress_research")

# === RESEARCHER TOOLS ===
async def researcher_tools(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        r_msgs = state.get("researcher_messages", [])
        if not r_msgs: return Command(goto="compress_research")
        last = r_msgs[-1]
        if not getattr(last, "tool_calls", None): return Command(goto="compress_research")
        tools = await get_all_tools(config)
        tbn = {t.name: t for t in tools if hasattr(t, "name")}
        obs = []
        for t in last.tool_calls:
            if t["name"] in tbn:
                try: obs.append(await tbn[t["name"]].ainvoke(t["args"], config))
                except: obs.append("[FALLBACK] Tool failed.")
            else: obs.append("[FALLBACK] Tool missing.")
        to = [ToolMessage(content=str(o), name=t["name"], tool_call_id=t["id"]) for o, t in zip(obs, last.tool_calls)]
        nc = [o for o in obs if isinstance(o, str)]
        ec = [m.content for m in r_msgs if hasattr(m, "content") and isinstance(m.content, str)]
        if check_information_satiation(nc, ec) or state.get("tool_call_iterations", 0) >= cfg.max_react_tool_calls:
            return Command(goto="compress_research", update={"researcher_messages": to})
        return Command(goto="researcher", update={"researcher_messages": to})
    except Exception as e:
        logging.error(f"researcher_tools failed: {e}")
        return Command(goto="compress_research")

# === COMPRESS RESEARCH (With JSON Resurrection + Distiller) ===
async def compress_research(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        mc = {"model": cfg.compression_model, "max_tokens": cfg.compression_model_max_tokens, "api_key": get_api_key_for_model(cfg.compression_model, config)}
        sm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(EvidenceGraphExtraction)
        r_msgs = state.get("researcher_messages", []) + [HumanMessage(content=compress_research_simple_human_message)]
        msgs = [SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))] + r_msgs
        msgs = _truncate(msgs)
        try:
            r = await sm.ainvoke(msgs)
            r_nodes = compute_epistemic_links(r.nodes)
            r.nodes = r_nodes
        except Exception as e:
            if "tool_use_failed" in str(e) or "400" in str(e):
                resurrected = _resurrect_json(str(e))
                if resurrected:
                    r = resurrected
                    r_nodes = compute_epistemic_links(r.nodes)
                    r.nodes = r_nodes
                else:
                    raise e
            else:
                raise e
        raw = "
".join([str(m.content) for m in filter_messages(r_msgs, include_types=["tool", "ai"])])
        aid = hashlib.md5(raw.encode()).hexdigest()[:8]
        rd = "Evidence:
" + "
".join([f"Fact {i+1}: {n.claim} ({n.url})" for i, n in enumerate(r.nodes)])
        for n in r.nodes:
            omega_local_memory.store(getattr(n, "claim", ""), getattr(n, "url", ""))
        return {"compressed_research": rd, "raw_notes": [raw], "evidence_graph": r.nodes, "artifact_id": aid, "executive_summary": rd[:500]}
    except Exception as e:
        logging.error(f"compress failed: {e}")
        return {"compressed_research": "Error", "raw_notes": [], "evidence_graph": [], "artifact_id": "err", "executive_summary": "Failed"}


rb = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
rb.add_node("researcher", researcher)
rb.add_node("researcher_tools", researcher_tools)
rb.add_node("compress_research", compress_research)
rb.add_edge(START, "researcher")
rb.add_edge("compress_research", END)
researcher_subgraph = rb.compile()

# === REASONING COUNCIL ===
async def reasoning_council(state, config):
    try:
        tier = state.get("complexity_tier", "Medium")
        if tier in ["Simple", "Medium"]:
            return Command(goto="adversarial_verification", update={"master_synthesis": "Standard inductive synthesis."})
        cfg = Configuration.from_runnable_config(config)
        argus = generate_argus_view(state.get("evidence_graph", []))
        findings = argus + "\n" + "\n".join(state.get("notes", []))[:6000]
        brief = state.get("research_brief", "")
        async def run_p(p):
            mc = {"model": cfg.research_model, "max_tokens": 2048, "api_key": get_api_key_for_model(cfg.research_model, config)}
            try:
                prompt = reasoning_council_prompt.format(paradigm=p, brief=brief, findings=findings[:10000])
                res = await safe_llm_invoke(init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]), [HumanMessage(content=prompt)])
                return f"### {p}\n{res.content}"
            except Exception: return f"### {p}\nSkipped."
        results = await asyncio.gather(*[run_p(p) for p in ["Deductive", "Inductive", "Abductive"]])
        return Command(goto="adversarial_verification", update={"master_synthesis": "\n\n".join(results)})
    except Exception as e:
        logging.error(f"council failed: {e}")
        return Command(goto="adversarial_verification", update={"master_synthesis": "Council failed."})

# === ADVERSARIAL VERIFICATION ===
async def adversarial_verification(state, config):
    try:
        ev = state.get("evidence_graph", [])
        ti = state.get("temporal_intent", "Current")
        vr = programmatic_epistemic_verification(ev, ti)
        argus = generate_argus_view(ev) if ev else ""
        return Command(goto="final_report_generation", update={"red_team_findings": vr["red_team_findings"], "devils_advocate_critique": vr["devils_advocate_critique"], "consensus_report": vr["consensus_report"] + "\n" + argus, "confidence_score": vr["confidence_score"]})
    except Exception as e:
        logging.error(f"verify failed: {e}")
        return Command(goto="final_report_generation", update={"confidence_score": 0.5})

# === FINAL REPORT (With ERC Circuit Breaker) ===
async def final_report_generation(state, config):
    try:
        conf = float(state.get("confidence_score", 0.0) or 0.0)
        contradictions = sum(1 for n in state.get("evidence_graph", []) if getattr(n, "contradicts", []))
        cfg = Configuration.from_runnable_config(config)
        if (conf < 0.65 or contradictions > 0) and state.get("research_iterations", 0) < cfg.max_researcher_iterations:
            return Command(goto="research_supervisor", update={"research_plan": add_targeted_research_nodes(state.get("evidence_graph", []), state.get("research_plan", [])), "complexity_tier": "Complex"})
        ev = state.get("evidence_graph", [])
        vn = []
        if ev:
            urls = [getattr(n, "url", "") for n in ev if getattr(n, "url", "")]
            if urls:
                try:
                    h = await validate_urls(urls)
                    ev = [n for n in ev if h.get(getattr(n, "url", ""), False)]
                except: pass
        verified = filter_and_verify_evidence(ev, temporal_intent=state.get("temporal_intent", "Current"))
        verified = await verify_citations_programmatically(verified)
        for n in verified:
            d = str(n.date_published) if getattr(n, "date_published", None) else "Unknown"
            vn.append(f"Fact: {getattr(n, 'claim', '')}\nSource: {getattr(n, 'url', '')}\nDate: {d}")
        notes = state.get("notes", []) + vn
        vfs = state.get("virtual_filesystem", {})
        ve = "\n\n".join([f"### VFS {k}\n{v}" for k, v in vfs.items()])
        findings = "\n".join(notes) + "\n\n" + ve
        wc = {"model": cfg.final_report_model, "max_tokens": cfg.final_report_model_max_tokens, "api_key": get_api_key_for_model(cfg.final_report_model, config)}
        prompt = final_report_generation_prompt.format(research_brief=state.get("research_brief", ""), findings=findings[:12000], date=get_today_str(), master_synthesis=state.get("master_synthesis", ""), consensus_report=state.get("consensus_report", "None"), confidence_score=state.get("confidence_score", 0.8), query_paradigm=state.get("query_paradigm", "General"))
        rep = await init_chat_model(model=wc["model"], max_tokens=wc["max_tokens"], api_key=wc["api_key"]).ainvoke([HumanMessage(content=prompt)])
        return {"final_report": rep.content, "messages": [rep], "notes": {"type": "override", "value": []}}
    except Exception as e:
        logging.error(f"report failed: {e}")
        return {"final_report": f"Fatal: {e}", "messages": [AIMessage(content="Failed")]}

# === META-LEARNING ===
async def meta_learning_node(state, config):
    try:
        conf = state.get("confidence_score", 0.8)
        iters = state.get("research_iterations", 0)
        if conf > 0.85 and iters < 4: return Command(goto=END)
        cfg = Configuration.from_runnable_config(config)
        mc = {"model": cfg.research_model, "max_tokens": 500, "api_key": get_api_key_for_model(cfg.research_model, config)}
        res = await safe_llm_invoke(init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]), [HumanMessage(content=meta_learning_prompt.format(confidence_score=conf, iterations=iters))])
        nl = state.get("lessons_learned", [])
        if "LESSON:" in res.content: nl.append(res.content.strip())
        return Command(goto=END, update={"lessons_learned": nl})
    except Exception: return Command(goto=END)

# === MAIN GRAPH COMPILATION ===
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

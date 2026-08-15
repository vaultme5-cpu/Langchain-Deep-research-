"""Omega Supremacy Engine (Groq-Only 9.9 Fabric)."""
import asyncio, hashlib, logging, re, json
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, get_buffer_string
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (clarify_with_user_instructions, compress_research_simple_human_message, compress_research_system_prompt, final_report_generation_prompt, lead_researcher_prompt, research_system_prompt, transform_messages_into_research_topic_prompt, meta_learning_prompt, reasoning_council_prompt, meta_cognitive_router_prompt)
from open_deep_research.state import (compute_epistemic_links, AgentInputState, AgentState, ClarifyWithUser, ConductResearch, EvidenceGraphExtraction, ResearchComplete, ResearcherOutputState, ResearcherState, ResearchQuestion, RouterDecision, SupervisorState, FinalReportArtifact)
from open_deep_research.utils import (check_information_satiation, filter_and_verify_evidence, get_all_tools, get_api_key_for_model, get_notes_from_tool_calls, get_today_str, validate_urls, think_tool, omega_local_memory, verify_citations_programmatically, calculate_epistemic_saturation, programmatic_epistemic_verification, _shield, omega_memory)
NL = chr(10)
GROQ_CONCURRENCY = 1
_BURST_SEMAPHORES = {}
_MODEL_TELEMETRY = []
_EXECUTION_HEALTH = {"status": "HEALTHY", "warnings": [], "failures": [], "fallbacks": []}
_RUN_BUDGET = {"used": 0.0, "cap": 24000.0}
_BRAIN_HEALTH = {}
def _current_loop():
    try: return asyncio.get_running_loop()
    except RuntimeError:
        try: return asyncio.get_event_loop()
        except RuntimeError: return asyncio.new_event_loop()
def _get_provider_semaphore(provider):
    key = (provider, id(_current_loop()))
    if key not in _BURST_SEMAPHORES:
        _BURST_SEMAPHORES[key] = asyncio.Semaphore(GROQ_CONCURRENCY if provider == "groq" else 2)
    return _BURST_SEMAPHORES[key]
def classify_model_error(e):
    s = str(e).lower()
    if "401" in s or "api key" in s or "unauthorized" in s: return "AUTH"
    if "403" in s or "permission" in s: return "PERMISSION"
    if "404" in s or "not found" in s or "no longer available" in s: return "MODEL_NOT_FOUND"
    if "429" in s or "rate limit" in s or "resource_exhausted" in s or "quota" in s: return "RATE_LIMIT"
    if "413" in s or "too long" in s or "context" in s or "maximum context" in s: return "CONTEXT_LIMIT"
    if "timeout" in s or "timed out" in s: return "TIMEOUT"
    if "500" in s or "502" in s or "503" in s or "overloaded" in s: return "SERVER_ERROR"
    if "400" in s or "invalid" in s or "tool_use_failed" in s: return "INVALID_REQUEST"
    return "UNKNOWN"
def _record_call(model_name, attempt, result, error_class):
    _MODEL_TELEMETRY.append({"model": model_name, "attempt": attempt, "result": result, "error_class": error_class})
def _record_health_event(component, kind, detail):
    if kind == "WARNING": _EXECUTION_HEALTH["warnings"].append(component + ": " + detail)
    elif kind == "FALLBACK": _EXECUTION_HEALTH["fallbacks"].append(component + ": " + detail)
    else: _EXECUTION_HEALTH["failures"].append(component + ": " + detail)
    if _EXECUTION_HEALTH["failures"]: _EXECUTION_HEALTH["status"] = "DEGRADED"
def _reset_run_state(cap=None):
    if cap is not None:
        _RUN_BUDGET["cap"] = max(
            1000.0,
            float(cap),
        )

    _RUN_BUDGET["used"] = 0.0
    _EXECUTION_HEALTH["status"] = "HEALTHY"
    _EXECUTION_HEALTH["warnings"] = []
    _EXECUTION_HEALTH["failures"] = []
    _EXECUTION_HEALTH["fallbacks"] = []
    _MODEL_TELEMETRY.clear()
def _budget_add(messages, resp):
    try:
        um = getattr(resp, "usage_metadata", None)
        inn = int(getattr(um, "input_tokens", 0) or 0)
        out = int(getattr(um, "output_tokens", 0) or 0)
        if inn == 0 and out == 0:
            inn = sum(len(str(getattr(m, "content", ""))) for m in messages) // 4
            out = len(str(getattr(resp, "content", ""))) // 4
        _RUN_BUDGET["used"] += float(inn + out)
    except Exception: pass
def _budget_left():
    return max(0.0, _RUN_BUDGET["cap"] - _RUN_BUDGET["used"])
def _brain_open(name, seconds, reason):
    import time as _t
    _BRAIN_HEALTH[name] = (_t.time() + float(seconds), str(reason))
    logging.error("brain locked: " + name + " for " + str(int(seconds)) + "s (" + reason + ")")
def _brain_is_open(name):
    import time as _t
    h = _BRAIN_HEALTH.get(name)
    if not h: return False
    if _t.time() >= h[0]:
        del _BRAIN_HEALTH[name]
        return False
    return True
def _parse_retry_seconds(err, default=300.0):
    m = re.search("again in ([0-9]+)m", err)
    if m: return float(m.group(1)) * 60.0
    m = re.search("retry in ([0-9]+)s", err)
    if m: return float(m.group(1))
    return default
def _lock_summary():
    import time as _t
    now = _t.time()
    parts = [k + " (" + str(int(max(0.0, v[0] - now) // 60)) + "m left)" for k, v in list(_BRAIN_HEALTH.items())]
    return "Locked: " + ", ".join(parts) if parts else "No live brain available."
def _chain_all_locked(cfg, kind):
    return all(_brain_is_open(n) for n, _ in _brain_chain(cfg, kind))
def _brain_chain(cfg, kind):
    intake = [(cfg.intake_model, cfg.intake_model_max_tokens)]
    research = [(cfg.research_model, cfg.research_model_max_tokens)]
    compress = [(cfg.compression_model, cfg.compression_model_max_tokens)]
    report = [(cfg.final_report_model, cfg.final_report_model_max_tokens)]

    reasoning_model = getattr(
        cfg,
        "reasoning_model",
        cfg.research_model,
    )
    reasoning_tokens = getattr(
        cfg,
        "reasoning_model_max_tokens",
        cfg.research_model_max_tokens,
    )
    reasoning = [(reasoning_model, reasoning_tokens)]

    if kind == "intake":
        return intake + research

    if kind == "compress":
        return compress + research + intake

    if kind == "report":
        return report + research + intake

    if kind == "reason":
        return reasoning + research + intake

    return research + intake


async def safe_llm_invoke(
    model,
    messages,
    max_attempts=4,
    brain_name=None,
    current_key=None,
    model_factory=None,
):
    sem = _get_provider_semaphore("groq")
    async with sem:
        last_error = None
        for attempt in range(max_attempts):
            try:
                resp = await model.ainvoke(messages)
                _budget_add(messages, resp)
                _record_call(brain_name, attempt, "SUCCESS", None)
                return resp
            except Exception as e:
                last_error = e
                cls = classify_model_error(e)
                _record_call(brain_name, attempt, "FAILED", cls)
                if cls == "RATE_LIMIT" and current_key:
                    new_key = _shield.get_key(last_failed=current_key)
                    if new_key and new_key != current_key:
                        current_key = new_key
                        if model_factory is not None:
                            try:
                                model = model_factory(new_key)
                            except Exception:
                                pass
                        else:
                            try:
                                model = model.with_config(
                                    {"api_key": new_key}
                                )
                            except Exception:
                                pass
                        _record_health_event(brain_name or "model", "WARNING", "key rotation")
                        await asyncio.sleep(1.0)
                        continue
                    raise RuntimeError("[EPISTEMIC FLAG]: key pool exhausted: " + str(e))
                if cls == "CONTEXT_LIMIT":
                    if len(messages) > 4:
                        messages = _truncate(messages[:2] + messages[-2:])
                        await asyncio.sleep(1.0)
                        continue
                    raise RuntimeError("[EPISTEMIC FLAG]: " + str(e))
                if cls in ("TIMEOUT", "SERVER_ERROR"):
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                if cls in ("AUTH", "PERMISSION", "MODEL_NOT_FOUND", "INVALID_REQUEST"):
                    raise RuntimeError("[EPISTEMIC FLAG]: " + str(e))
                if attempt == max_attempts - 1:
                    raise RuntimeError("[EPISTEMIC FLAG]: " + str(e))
                await asyncio.sleep(2.0 * (attempt + 1))
        raise RuntimeError("[EPISTEMIC FLAG]: Max retries exhausted. " + str(last_error))
async def _brain_invoke(cfg, config, kind, messages, structured=None, tools=None):
    chain = _brain_chain(cfg, kind)
    if _budget_left() <= 0.0:
        raise RuntimeError("[EPISTEMIC FLAG]: Run token budget exhausted.")
    last = None
    for name, tok in chain:
        if _brain_is_open(name):
            continue
        key = get_api_key_for_model(name, config)
        try:
            def build_model(selected_key):
                built = init_chat_model(
                    model=name,
                    max_tokens=tok,
                    api_key=selected_key,
                )

                if structured is not None:
                    built = built.with_structured_output(structured)

                if tools is not None:
                    built = built.bind_tools(tools)

                return built

            m = build_model(key)

            return await safe_llm_invoke(
                m,
                messages,
                max_attempts=max(
                    1,
                    int(getattr(
                        cfg,
                        "max_rate_limit_retries",
                        4,
                    )) + 1,
                ),
                brain_name=name,
                current_key=key,
                model_factory=build_model,
            )
        except Exception as e:
            last = e
            cls = classify_model_error(e)
            if cls in ("RATE_LIMIT", "MODEL_NOT_FOUND", "SERVER_ERROR", "AUTH", "PERMISSION"):
                _brain_open(name, _parse_retry_seconds(str(e)) if cls == "RATE_LIMIT" else 21600.0, cls)
                _record_health_event(name, "FALLBACK", cls)
                logging.error("brain failover from " + name + " (" + cls + ")")
                continue
            raise
    raise RuntimeError("[EPISTEMIC FLAG]: All brains exhausted or locked. " + str(last) + " | " + _lock_summary())
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
def _resurrect_json(raw_text):
    try:
        text = str(raw_text or "").replace("```json", "").replace("```", "")
        text = re.sub(r"<function.*?>", "", text, flags=re.DOTALL)
        text = re.sub(r"</function.*?>", "", text, flags=re.DOTALL)
        start = -1
        for i, ch in enumerate(text):
            if ch in "{[": start = i; break
        if start == -1: return None
        text = text[start:]
        if text.startswith("["): text = '{"nodes": ' + text + '}'
        text = re.sub(r",\s*([}\]])", r"\1", text)
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
    return hashlib.sha256((ev_repr + " || " + plan_repr + " || " + comp_repr).encode("utf-8")).hexdigest()
def generate_argus_view(nodes):
    if not nodes: return "No structured evidence."
    sc = {getattr(n, "citation_index", 0): 0 for n in nodes}
    for n in nodes:
        for s in getattr(n, "supports", []):
            if s in sc: sc[s] += 1
    view = "### ARGUS VIEW" + NL
    for n in [n for n in nodes if sc.get(getattr(n, "citation_index", 0), 0) >= 2][:5]:
        view += "- [" + str(getattr(n, "citation_index", 0)) + "] " + str(getattr(n, "claim", "")) + " (x" + str(sc.get(getattr(n, "citation_index", 0), 0)) + ")" + NL
    for n in [n for n in nodes if getattr(n, "contradicts", [])][:3]:
        view += "- [" + str(getattr(n, "citation_index", 0)) + "] " + str(getattr(n, "claim", "")) + " (CONTRADICTS)" + NL
    return view
def add_targeted_research_nodes(evidence_graph, research_plan):
    plan = list(research_plan) if research_plan else []
    existing = {n.get("node_id") for n in plan if isinstance(n, dict)}
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
        _reset_run_state(
            getattr(
                cfg,
                "run_token_budget",
                _RUN_BUDGET["cap"],
            )
        )
        if not cfg.allow_clarification: return Command(goto="write_research_brief")
        if _chain_all_locked(cfg, "intake") and _chain_all_locked(cfg, "work"):
            return Command(goto=END, update={"messages": [AIMessage(content="All research brains are quota-locked. " + _lock_summary() + " Wait for the rolling window, then retry.")], "final_report": "Capacity locked. " + _lock_summary()})
        prompt = clarify_with_user_instructions.format(messages=get_buffer_string(state.get("messages", [])), date=get_today_str())
        r = await _brain_invoke(cfg, config, "intake", [HumanMessage(content=prompt)], structured=ClarifyWithUser)
        if getattr(r, "need_clarification", False): return Command(goto=END, update={"messages": [AIMessage(content=getattr(r, "question", "Please clarify."))]})
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=getattr(r, "verification", "Proceeding."))]})
    except Exception as e:
        logging.error("clarify failed: " + str(e))
        return Command(goto="write_research_brief")
async def write_research_brief(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        prompt = transform_messages_into_research_topic_prompt.format(messages=get_buffer_string(state.get("messages", [])), date=get_today_str())
        r = await _brain_invoke(cfg, config, "intake", [HumanMessage(content=prompt)], structured=ResearchQuestion)
        mem = omega_memory.get_context_prompt()
        sup_sys = lead_researcher_prompt.format(date=get_today_str(), mcp_prompt=cfg.mcp_prompt or " ", max_concurrent_research_units=cfg.max_concurrent_research_units, max_researcher_iterations=cfg.max_researcher_iterations, temporal_intent=getattr(r, "temporal_intent", "Current"), complexity_tier="Pending", lessons_learned=mem, hard_constraints=getattr(r, "hard_constraints", []), memory_context=mem)
        return Command(goto="meta_cognitive_router", update={"research_brief": getattr(r, "research_brief", ""), "temporal_intent": getattr(r, "temporal_intent", "Current"), "hard_constraints": getattr(r, "hard_constraints", []), "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=getattr(r, "research_brief", ""))]}})
    except Exception as e:
        logging.error("brief failed: " + str(e))
        return Command(goto=END)
async def meta_cognitive_router(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        mem = omega_memory.get_context_prompt()
        prompt = meta_cognitive_router_prompt.format(research_brief=state.get("research_brief", ""), date=get_today_str(), memory_context=mem)
        r = await _brain_invoke(cfg, config, "intake", [HumanMessage(content=prompt)], structured=RouterDecision)
        sup_sys = lead_researcher_prompt.format(date=get_today_str(), mcp_prompt=cfg.mcp_prompt or " ", max_concurrent_research_units=getattr(r, "dynamic_research_units", cfg.max_concurrent_research_units), max_researcher_iterations=getattr(r, "dynamic_tool_budget", cfg.max_researcher_iterations), complexity_tier=getattr(r, "complexity_tier", "Medium"), temporal_intent=state.get("temporal_intent", "Current"), lessons_learned=mem, hard_constraints=state.get("hard_constraints", []), memory_context=mem)
        pd = []
        for n in (getattr(r, "research_plan", []) or []):
            try: pd.append(n.model_dump())
            except Exception: pd.append({"node_id": str(getattr(n, "node_id", "")), "topic": str(getattr(n, "topic", "")), "depends_on": list(getattr(n, "depends_on", []))})
        return Command(goto="research_supervisor", update={"query_paradigm": getattr(r, "query_paradigm", "General"), "complexity_tier": getattr(r, "complexity_tier", "Medium"), "dynamic_tool_budget": getattr(r, "dynamic_tool_budget", cfg.max_react_tool_calls), "dynamic_research_units": getattr(r, "dynamic_research_units", cfg.max_concurrent_research_units), "research_plan": pd, "completed_nodes": [], "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=state.get("research_brief", ""))]}})
    except Exception as e:
        logging.error("router failed: " + str(e))
        return Command(goto=END)
async def researcher(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        tools = await get_all_tools(config)
        mem = omega_memory.get_context_prompt()
        prompt = research_system_prompt.format(mcp_prompt=cfg.mcp_prompt or " ", date=get_today_str(), temporal_intent=state.get("temporal_intent", "Current"), hard_constraints=state.get("hard_constraints", []), memory_context=mem)
        r_msgs = state.get("researcher_messages", [])
        core = [m for m in r_msgs if isinstance(m, (SystemMessage, HumanMessage))]
        recent = [m for m in r_msgs if not isinstance(m, (SystemMessage, HumanMessage))][-6:]
        msgs = _truncate([SystemMessage(content=prompt)] + core + recent, getattr(cfg, "max_tool_payload_chars", 5500))
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
        to = [ToolMessage(content="[UNTRUSTED DATA] " + str(o), name=str(t.get("name", "tool")), tool_call_id=str(t.get("id", "tool"))) for o, t in zip(obs, calls)]
        nc = [str(o) for o in obs if isinstance(o, str)]
        ec = [
            str(getattr(m, "content", ""))
            for m in r_msgs
            if isinstance(
                getattr(m, "content", ""),
                str,
            )
        ]

        max_calls = int(
            state.get(
                "dynamic_tool_budget",
                cfg.max_react_tool_calls,
            )
            or cfg.max_react_tool_calls
        )

        if (
            check_information_satiation(nc, ec)
            or int(
                state.get(
                    "tool_call_iterations",
                    0,
                )
                or 0
            ) >= max_calls
        ):
            return Command(goto="compress_research", update={"researcher_messages": to})
        return Command(goto="researcher", update={"researcher_messages": to})
    except Exception as e:
        logging.error("researcher_tools failed: " + str(e))
        return Command(goto="compress_research")
async def compress_research(state, config):
    try:
        cfg = Configuration.from_runnable_config(config)
        r_msgs = state.get("researcher_messages", [])
        sys_msg = SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))
        tool_msgs = [m for m in r_msgs if isinstance(m, ToolMessage)]
        tool_text = NL.join([str(m.content) for m in tool_msgs])
        chunk_limit = int(getattr(cfg, "max_compression_chunk_chars", 2600))
        all_nodes = []
        if len(tool_text) > chunk_limit * 2:
            chunks_all = _chunk_text(tool_text, chunk_limit)
            chunks = list(dict.fromkeys([chunks_all[i * (len(chunks_all) - 1) // 3] for i in range(4)])) if len(chunks_all) > 3 else chunks_all
            for c in chunks:
                msgs = [sys_msg, HumanMessage(content="Extract facts from this research chunk." + NL + c), HumanMessage(content=compress_research_simple_human_message)]
                try:
                    r = await _brain_invoke(cfg, config, "compress", msgs, structured=EvidenceGraphExtraction)
                    if getattr(r, "nodes", None): all_nodes.extend(r.nodes)
                except Exception as ce:
                    rescued = _resurrect_json(str(ce))
                    if rescued and getattr(rescued, "nodes", None): all_nodes.extend(rescued.nodes)
        if not all_nodes:
            msgs = _truncate([sys_msg] + _chunk_messages_for_compression(r_msgs, chunk_limit) + [HumanMessage(content=compress_research_simple_human_message)], getattr(cfg, "max_tool_payload_chars", 5500))
            try:
                r = await _brain_invoke(cfg, config, "compress", msgs, structured=EvidenceGraphExtraction)
                all_nodes = getattr(r, "nodes", []) or []
            except Exception as e:
                rescued = _resurrect_json(str(e))
                if rescued and getattr(rescued, "nodes", None): all_nodes = rescued.nodes
                else: raise e
        r_nodes = [n for n in compute_epistemic_links(all_nodes) if len(str(getattr(n, "claim", "")).strip()) >= 40]
        aid = hashlib.sha256((tool_text or "none").encode("utf-8")).hexdigest()[:10]
        rd = "Evidence:" + NL + NL.join(["Fact " + str(i+1) + ": " + str(getattr(n, "claim", "")) + " (" + str(getattr(n, "url", "")) + ")" for i, n in enumerate(r_nodes)]) if r_nodes else "No evidence extracted."
        for n in r_nodes: omega_local_memory.store(getattr(n, "claim", ""), getattr(n, "url", ""))
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
        no_progress = no_progress + 1 if prev == snapshot else 0
        erc_update = {"erc_frontier_fingerprint": snapshot, "erc_no_progress_count": no_progress, "research_iterations": iters + 1}
        sat = calculate_epistemic_saturation(state.get("evidence_graph", []), state.get("research_plan", []))
        if sat >= 0.85 or iters >= cfg.max_researcher_iterations:
            erc_update["supervisor_messages"] = [AIMessage(content="", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "halt"}])]
            return Command(goto="supervisor_tools", update=erc_update)
        if no_progress >= cfg.erc_max_stagnation_iterations:
            if iters + 1 < cfg.max_researcher_iterations:
                erc_update["research_plan"] = add_targeted_research_nodes(state.get("evidence_graph", []), state.get("research_plan", []))
                erc_update["supervisor_messages"] = [AIMessage(content="", tool_calls=[{"name": "ConductResearch", "args": {"node_id": "FB_ERC", "research_topic": "Diversify search and resolve stagnant evidence. Focus on contradictions, missing dates, and source diversity."}, "id": "erc"}])]
                return Command(goto="supervisor_tools", update=erc_update)
            erc_update["supervisor_messages"] = [AIMessage(content="", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "halt"}])]
            return Command(goto="supervisor_tools", update=erc_update)
        sup_msgs = list(state.get("supervisor_messages", []))
        core = [m for m in sup_msgs if isinstance(m, (SystemMessage, HumanMessage)) and "DAG_STATUS" not in str(getattr(m, "content", ""))]
        recent = [m for m in sup_msgs if not isinstance(m, (SystemMessage, HumanMessage))][-4:]
        sup_msgs = core + recent
        if state.get("research_plan"):
            sup_msgs.append(SystemMessage(content=NL + "<DAG_STATUS>" + NL + "Plan: " + json.dumps(state.get("research_plan", []), default=str)[:3000] + NL + "Completed: " + str(state.get("completed_nodes", []))[:1000] + NL + "</DAG_STATUS>"))
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
        completed = set(state.get("completed_nodes", []))
        plan_dict = {n.get("node_id"): n for n in state.get("research_plan", []) if isinstance(n, dict)}
        valid_cc = []
        blocked_atm = []
        for t in cc:
            nid = str(
                (t.get("args", {}) or {}).get("node_id")
                or ""
            )

            if nid not in plan_dict:
                blocked_atm.append(
                    ToolMessage(
                        content=(
                            "REJECTED: Unknown DAG node "
                            + nid
                            + ". Only declared research "
                            + "nodes may execute."
                        ),
                        name=str(
                            t.get(
                                "name",
                                "ConductResearch",
                            )
                        ),
                        tool_call_id=str(
                            t.get("id", "tool")
                        ),
                    )
                )
                continue

            deps = plan_dict[nid].get(
                "depends_on",
                [],
            )

            if all(
                str(dep) in completed
                for dep in deps
            ):
                valid_cc.append(t)
            else:
                blocked_atm.append(
                    ToolMessage(
                        content=(
                            "BLOCKED: Node "
                            + nid
                            + " dependencies "
                            + str(deps)
                            + " not met."
                        ),
                        name=str(
                            t.get(
                                "name",
                                "ConductResearch",
                            )
                        ),
                        tool_call_id=str(
                            t.get("id", "tool")
                        ),
                    )
                )
        cc = valid_cc
        if not cc and not blocked_atm:
            tm = [ToolMessage(content="Acknowledged.", name=str(t.get("name", "tool")), tool_call_id=str(t.get("id", "tool"))) for t in calls]
            return Command(goto="supervisor", update={"supervisor_messages": tm or [AIMessage(content="Continuing.")]})
        if not cc and blocked_atm:
            return Command(goto="supervisor", update={"supervisor_messages": blocked_atm})
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
        for obs, t in zip(results, allowed):
            nid = str((t.get("args", {}) or {}).get("node_id") or "")
            if not nid or isinstance(obs, Exception): continue
            comp_res = str((obs or {}).get("compressed_research", ""))
            if comp_res in ("", "Error", "No evidence extracted."): continue
            ev_graph = (obs or {}).get("evidence_graph", [])
            if not ev_graph: continue
            nc.append(nid)
        if nc: up["completed_nodes"] = list(set(list(state.get("completed_nodes", [])) + nc))
        up["supervisor_messages"] = atm + blocked_atm
        return Command(goto="supervisor", update=up)
    except Exception as e:
        logging.error("supervisor_tools failed: " + str(e))
        return Command(goto=END)
sb = StateGraph(SupervisorState, config_schema=Configuration)
sb.add_node("supervisor", supervisor)
sb.add_node("supervisor_tools", supervisor_tools)
sb.add_edge(START, "supervisor")
sb.add_conditional_edges("supervisor", lambda s: "supervisor_tools" if s.get("supervisor_messages") and getattr(s["supervisor_messages"][-1], "tool_calls", None) else END)
sb.add_edge("supervisor_tools", "supervisor")
supervisor_subgraph = sb.compile()

async def reasoning_council(state, config):
    try:
        if str(state.get("complexity_tier", "Medium")) != "Expert":
            return Command(goto="adversarial_verification", update={"master_synthesis": "Standard inductive synthesis."})
        cfg = Configuration.from_runnable_config(config)
        argus = generate_argus_view(state.get("evidence_graph", []))
        findings = argus + NL + NL.join([str(x) for x in state.get("notes", [])])[:6000]
        brief = state.get("research_brief", "")
        async def run_p(p):
            try:
                prompt = reasoning_council_prompt.format(paradigm=p, brief=brief, findings=findings[:10000])
                res = await _brain_invoke(
                    cfg,
                    config,
                    "reason",
                    [HumanMessage(content=prompt)],
                )
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
        vr = programmatic_epistemic_verification(ev, state.get("temporal_intent", "Current"))
        argus = generate_argus_view(ev) if ev else ""
        return Command(goto="final_report_generation", update={"red_team_findings": vr.get("red_team_findings", ""), "devils_advocate_critique": vr.get("devils_advocate_critique", ""), "consensus_report": str(vr.get("consensus_report", "")) + NL + argus, "confidence_score": float(vr.get("confidence_score", 0.5))})
    except Exception as e:
        logging.error("verify failed: " + str(e))
        return Command(goto="final_report_generation", update={"confidence_score": 0.5})
def _sanitize_report_citations(content, evidence_count):
    text = str(content or "")

    def replace_invalid(match):
        n = int(match.group(1))
        return match.group(0) if 1 <= n <= evidence_count else ""

    text = re.sub(r"\[(\d+)\]", replace_invalid, text)

    if "Sources" not in text:
        text += NL + NL + "Sources" + NL

    return text.strip()



def _render_final_report(
    artifact,
    verified,
    confidence,
    consensus,
):
    valid_ids = set(
        range(
            1,
            len(verified) + 1,
        )
    )

    def refs(ids):
        result = []

        for item in ids or []:
            try:
                number = int(item)
            except Exception:
                continue

            if (
                number in valid_ids
                and number not in result
            ):
                result.append(number)

        return "".join(
            " [" + str(x) + "]"
            for x in result
        )

    lines = []

    title = str(
        getattr(
            artifact,
            "title",
            "",
        )
        or "Omega Research Report"
    ).strip()

    lines.append("# " + title)

    summary = str(
        getattr(
            artifact,
            "executive_summary",
            "",
        )
        or ""
    ).strip()

    if summary:
        lines.extend([
            "",
            "## Executive Summary",
            summary
            + refs(
                getattr(
                    artifact,
                    "executive_evidence_ids",
                    [],
                )
            ),
        ])

    sections = getattr(
        artifact,
        "sections",
        [],
    ) or []

    for section in sections:
        heading = str(
            getattr(
                section,
                "heading",
                "",
            )
            or "Analysis"
        ).strip()

        content = str(
            getattr(
                section,
                "content",
                "",
            )
            or ""
        ).strip()

        if not content:
            continue

        lines.extend([
            "",
            "## " + heading,
            content
            + refs(
                getattr(
                    section,
                    "evidence_ids",
                    [],
                )
            ),
        ])

    lines.extend([
        "",
        "## Sources",
    ])

    for index, node in enumerate(
        verified,
        start=1,
    ):
        title = str(
            getattr(
                node,
                "title",
                "",
            )
            or "Source"
        )

        url = str(
            getattr(
                node,
                "url",
                "",
            )
        )

        lines.append(
            "["
            + str(index)
            + "] "
            + title
            + " — "
            + url
        )

    lines.extend([
        "",
        "## Epistemic Audit",
        str(consensus or "N/A"),
        "Confidence: "
        + str(
            round(
                float(confidence),
                3,
            )
        ),
        "",
        "## Watchlist",
    ])

    watchlist = getattr(
        artifact,
        "watchlist",
        [],
    ) or []

    if watchlist:
        for item in watchlist:
            if str(item).strip():
                lines.append(
                    "- " + str(item).strip()
                )
    else:
        lines.append(
            "- No watchlist items."
        )

    return "\n".join(lines).strip()


async def final_report_generation(state, config):
    try:
        conf = float(state.get("confidence_score", 0.0) or 0.0)
        contradictions = sum(1 for n in state.get("evidence_graph", []) if getattr(n, "contradicts", []))
        cfg = Configuration.from_runnable_config(config)
        iters = int(state.get("research_iterations", 0) or 0)
        if (conf < cfg.min_final_confidence or contradictions > 0) and iters < cfg.max_researcher_iterations:
            return Command(goto="research_supervisor", update={"research_plan": add_targeted_research_nodes(state.get("evidence_graph", []), state.get("research_plan", [])), "complexity_tier": "Complex"})
        ev = state.get("evidence_graph", [])
        verified = []
        weak_ids = set()
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
                if isinstance(checked, dict):
                    weak_ids = set(id(w) for w in checked.get("weak", []))
                    verified = checked.get("strong", []) + checked.get("weak", [])
                elif checked:
                    verified = checked
            except Exception: pass
        vn = []
        for n in verified:
            d = str(getattr(n, "date_published", "")) if getattr(n, "date_published", None) else "Unknown"
            vn.append("[" + str(len(vn) + 1) + "] " + ("[UNVERIFIED] " if id(n) in weak_ids else "") + str(getattr(n, "claim", "")) + NL + "Source: " + str(getattr(n, "url", "")) + NL + "Date: " + d)
        # Citation numbers must map ONLY to verified evidence.
        # Free-form notes are intentionally excluded from the numbered findings
        # so the report cannot accidentally cite an internal note as a source.
        vfs = state.get("virtual_filesystem", {})
        ve = (NL + NL).join(
            ["### VFS " + str(k) + NL + str(v) for k, v in vfs.items()]
        )
        findings = NL.join(vn) + NL + NL + ve
        prompt = final_report_generation_prompt.format(research_brief=state.get("research_brief", ""), findings=findings[:12000], date=get_today_str(), master_synthesis=state.get("master_synthesis", ""), consensus_report=state.get("consensus_report", "None"), confidence_score=state.get("confidence_score", 0.8), query_paradigm=state.get("query_paradigm", "General"))
        artifact = await _brain_invoke(
            cfg,
            config,
            "report",
            [HumanMessage(content=prompt)],
            structured=FinalReportArtifact,
        )

        content = _render_final_report(
            artifact,
            verified,
            state.get(
                "confidence_score",
                0.0,
            ),
            state.get(
                "consensus_report",
                "N/A",
            ),
        )

        if verified:
            sources = NL + NL + "Sources" + NL
            for idx, node in enumerate(verified, start=1):
                sources += (
                    "[" + str(idx) + "] "
                    + str(getattr(node, "title", "") or "Source")
                    + " — "
                    + str(getattr(node, "url", ""))
                    + NL
                )
            if "Epistemic Audit" in content:
                content = content.split("Epistemic Audit", 1)[0].rstrip()
            content += sources
            content += NL + "Epistemic Audit"
            content += NL + str(state.get("consensus_report", "N/A"))
            content += NL + "Confidence: " + str(state.get("confidence_score", 0.0))
            content += NL + "Watchlist"
            content += NL + str(state.get("devils_advocate_critique", "N/A"))

        return {
            "final_report": content,
            "messages": [rep],
            "notes": {"type": "override", "value": []}
        }
    except Exception as e:
        logging.error("report failed: " + str(e))
        return {"final_report": "Fatal: " + str(e), "messages": [AIMessage(content="Failed")]}
async def meta_learning_node(state, config):
    try:
        conf = float(state.get("confidence_score", 0.8) or 0.8)
        iters = int(state.get("research_iterations", 0) or 0)
        if conf > 0.85 and iters < 4: return Command(goto=END)
        cfg = Configuration.from_runnable_config(config)
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

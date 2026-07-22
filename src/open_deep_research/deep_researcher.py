"""Main LangGraph implementation for Project Omega (Ghost Eradicated)."""
import asyncio, logging, hashlib, re
from typing import Literal
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
    lead_researcher_prompt, research_system_prompt, transform_messages_into_research_topic_prompt,
    meta_learning_prompt, reasoning_council_prompt, red_team_prompt,
    devils_advocate_prompt, consensus_builder_prompt, meta_cognitive_router_prompt,
)
from open_deep_research.state import (
    AgentInputState, AgentState, ClarifyWithUser, ConductResearch,
    EvidenceGraphExtraction, ResearchComplete, ResearcherOutputState,
    ResearcherState, ResearchQuestion, RouterDecision, SupervisorState,
)
from open_deep_research.utils import (
    check_information_satiation, filter_and_verify_evidence,
    anthropic_websearch_called, get_all_tools, get_api_key_for_model,
    get_model_token_limit, get_notes_from_tool_calls, get_today_str,
    is_token_limit_exceeded, openai_websearch_called,
    remove_up_to_last_ai_message, validate_urls, think_tool, compile_search_results,
)

configurable_model = init_chat_model(configurable_fields=("model", "max_tokens", "api_key"))

async def safe_llm_invoke(model, messages):
    for attempt in range(3):
        try: return await model.ainvoke(messages)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e).lower():
                await asyncio.sleep(1.5 * (attempt + 1)); continue
            if attempt == 2: raise RuntimeError("[EPISTEMIC FLAG]: LLM constraint.")
    raise RuntimeError("[EPISTEMIC FLAG]: LLM constraint.")

async def clarify_with_user(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification: return Command(goto="write_research_brief")
    messages = state["messages"]
    mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    cm = configurable_model.with_structured_output(ClarifyWithUser).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(mc)
    response = await cm.ainvoke([HumanMessage(content=clarify_with_user_instructions.format(messages=get_buffer_string(messages), date=get_today_str()))])
    if response.need_clarification: return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})

async def write_research_brief(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    rm = configurable_model.with_structured_output(ResearchQuestion).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(mc)
    response = await rm.ainvoke([HumanMessage(content=transform_messages_into_research_topic_prompt.format(messages=get_buffer_string(state.get("messages", [])), date=get_today_str()))])
    sup_sys = lead_researcher_prompt.format(date=get_today_str(), mcp_prompt=configurable.mcp_prompt or "", max_concurrent_research_units=configurable.max_concurrent_research_units, max_researcher_iterations=configurable.max_researcher_iterations, temporal_intent=getattr(response, "temporal_intent", "Current"), complexity_tier="Pending", lessons_learned="
".join(state.get("lessons_learned", [])))
    return Command(goto="meta_cognitive_router", update={"research_brief": response.research_brief, "temporal_intent": getattr(response, "temporal_intent", "Current"), "hard_constraints": getattr(response, "hard_constraints", []), "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=response.research_brief)]}})

async def meta_cognitive_router(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    mc = {"model": configurable.research_model, "max_tokens": 4096, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    rm = configurable_model.with_structured_output(RouterDecision).with_config(mc)
    response = await rm.ainvoke([HumanMessage(content=meta_cognitive_router_prompt.format(research_brief=state.get("research_brief", ""), date=get_today_str()))])
    sup_sys = lead_researcher_prompt.format(date=get_today_str(), mcp_prompt=configurable.mcp_prompt or "", max_concurrent_research_units=response.dynamic_research_units, max_researcher_iterations=response.dynamic_tool_budget, complexity_tier=response.complexity_tier, temporal_intent=state.get("temporal_intent", "Current"), lessons_learned="
".join(state.get("lessons_learned", [])))
    plan_dicts = [n.dict() for n in response.research_plan] if hasattr(response, "research_plan") and response.research_plan else []
    return Command(goto="research_supervisor", update={"query_paradigm": response.query_paradigm, "complexity_tier": response.complexity_tier, "dynamic_tool_budget": response.dynamic_tool_budget, "dynamic_research_units": response.dynamic_research_units, "research_plan": plan_dicts, "completed_nodes": [], "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=state.get("research_brief", ""))])

async def supervisor(state: SupervisorState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    rm = configurable_model.bind_tools([ConductResearch, ResearchComplete, think_tool]).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(mc)
    sup_msgs = state.get("supervisor_messages", [])
    plan = state.get("research_plan", [])
    completed = state.get("completed_nodes", [])
    if plan:
        dag_status = f"\n<DAG_STATUS>\nPlan: {plan}\nCompleted Nodes: {completed}\n</DAG_STATUS>"
        if not any("DAG_STATUS" in str(m.content) for m in sup_msgs if hasattr(m, 'content')):
            sup_msgs = sup_msgs + [SystemMessage(content=dag_status)]
    response = await rm.ainvoke(sup_msgs)
    return Command(goto="supervisor_tools", update={"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1})

async def supervisor_tools(state: SupervisorState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    sup_msgs = state.get("supervisor_messages", [])
    iters = state.get("research_iterations", 0)
    last_msg = sup_msgs[-1]
    if iters > configurable.max_researcher_iterations or not last_msg.tool_calls or any(tc["name"] == "ResearchComplete" for tc in last_msg.tool_calls):
        return Command(goto=END, update={"notes": get_notes_from_tool_calls(sup_msgs), "research_brief": state.get("research_brief", "")})
    all_tool_msgs = []
    update_payload = {"supervisor_messages": []}
    for tc in [t for t in last_msg.tool_calls if t["name"] == "think_tool"]:
        all_tool_msgs.append(ToolMessage(content="Reflection recorded: " + tc["args"]["reflection"], name="think_tool", tool_call_id=tc["id"]))
    conduct_calls = [t for t in last_msg.tool_calls if t["name"] == "ConductResearch"]
    if conduct_calls:
        allowed = conduct_calls[:configurable.max_concurrent_research_units]
        overflow = conduct_calls[configurable.max_concurrent_research_units:]
        try:
            tasks = [researcher_subgraph.ainvoke({"researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])], "research_topic": tc["args"]["research_topic"]}, config) for tc in allowed]
            results = await asyncio.gather(*tasks)
            artifacts_update = {}
            agg_graph = []
            for obs, tc in zip(results, allowed):
                art_id = obs.get("artifact_id", tc["id"])
                summary = obs.get("executive_summary", obs.get("compressed_research", "Error"))
                artifacts_update[art_id] = obs.get("compressed_research", "")
                all_tool_msgs.append(ToolMessage(content=f"ARTIFACT ID: {art_id}\nEXECUTIVE SUMMARY:\n{summary}", name=tc["name"], tool_call_id=tc["id"]))
                agg_graph.extend(obs.get("evidence_graph", []))
            for tc in overflow:
                all_tool_msgs.append(ToolMessage(content="Error: Exceeded max concurrent units.", name="ConductResearch", tool_call_id=tc["id"]))
            if artifacts_update: update_payload["research_artifacts"] = artifacts_update
            if agg_graph: update_payload["evidence_graph"] = agg_graph
            newly_completed = [tc["args"].get("node_id", "") for tc in allowed if tc["args"].get("node_id")]
            if newly_completed:
                current_completed = set(state.get("completed_nodes", []))
                update_payload["completed_nodes"] = list(current_completed.union(set(newly_completed)))
            raw_concat = "\n".join(["\n".join(obs.get("raw_notes", [])) for obs in results])
            if raw_concat: update_payload["raw_notes"] = [raw_concat]
        except Exception as e:
            return Command(goto=END, update={"notes": get_notes_from_tool_calls(sup_msgs), "research_brief": state.get("research_brief", "")})
    update_payload["supervisor_messages"] = all_tool_msgs
    return Command(goto="supervisor", update=update_payload)

supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    tools = await get_all_tools(config)
    if not tools: raise ValueError("No tools found.")
    mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    prompt = research_system_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_str()
    rm = configurable_model.bind_tools(tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(mc)
    msgs = [SystemMessage(content=prompt)] + state.get("researcher_messages", [])
    response = await rm.ainvoke(msgs)
    return Command(goto="researcher_tools", update={"researcher_messages": [response], "tool_call_iterations": state.get("tool_call_iterations", 0) + 1})

async def execute_tool_safely(tool, args, config):
    for attempt in range(3):
        try: return await tool.ainvoke(args, config)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e).lower(): await asyncio.sleep(2 ** attempt); continue
            if attempt == 2: return f"[TOOL FALLBACK]: {getattr(tool, 'name', 'unknown')} failed."
    return "[TOOL FALLBACK]: Max retries exceeded."

async def researcher_tools(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    r_msgs = state.get("researcher_messages", [])
    last_msg = r_msgs[-1]
    if not last_msg.tool_calls and not (openai_websearch_called(last_msg) or anthropic_websearch_called(last_msg)):
        return Command(goto="compress_research")
    tools = await get_all_tools(config)
    tools_by_name = {t.name if hasattr(t, "name") else t.get("name", "web_search"): t for t in tools}
    obs = await asyncio.gather(*[execute_tool_safely(tools_by_name[tc["name"]], tc["args"], config) for tc in last_msg.tool_calls])
    compiled_obs = []
    for o, tc in zip(obs, last_msg.tool_calls):
        if isinstance(o, str) and tc["name"] in ["github_sniper", "huggingface_sniper", "hackernews_sniper", "semantic_scholar_search"]:
            compiled_obs.append(compile_search_results(tc["name"], o))
        else: compiled_obs.append(o)
    tool_outputs = [ToolMessage(content=o, name=tc["name"], tool_call_id=tc["id"]) for o, tc in zip(compiled_obs, last_msg.tool_calls)]
    new_claims = [o for o in compiled_obs if isinstance(o, str)]
    existing_context = [m.content for m in r_msgs if hasattr(m, 'content') and isinstance(m.content, str)]
    if check_information_satiation(new_claims, existing_context): return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
    if state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls or any(tc["name"] == "ResearchComplete" for tc in last_msg.tool_calls): return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})

async def compress_research(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    sm = configurable_model.with_config({"model": configurable.compression_model, "max_tokens": configurable.compression_model_max_tokens, "api_key": get_api_key_for_model(configurable.compression_model, config), "tags": ["langsmith:nostream"]})
    r_msgs = state.get("researcher_messages", [])
    r_msgs.append(HumanMessage(content=compress_research_simple_human_message))
    attempts = 0
    while attempts < 3:
        try:
            msgs = [SystemMessage(content=compress_research_system_prompt.format(date=get_today_str())] + r_msgs
            response = await sm.with_structured_output(EvidenceGraphExtraction).ainvoke(msgs)
            raw_notes = "\n".join([str(m.content) for m in filter_messages(r_msgs, include_types=["tool", "ai"])])
            readable = "Extracted Evidence Graph:\n"
            for i, node in enumerate(response.nodes):
                sup = getattr(node, 'supports', [])
                con = getattr(node, 'contradicts', [])
                readable += f"Fact {i+1}: {getattr(node, 'claim', '')}\nSource: {getattr(node, 'title', '')} ({getattr(node, 'url', '')})\nSupports: {sup} | Contradicts: {con}\n"
            art_id = hashlib.md5(raw_notes.encode()).hexdigest()[:8]
            exec_sum = readable[:800] + f"...\n[Full graph in Artifact {art_id}]"
            return {"compressed_research": readable, "raw_notes": [raw_notes], "evidence_graph": response.nodes, "artifact_id": art_id, "executive_summary": exec_sum}
        except Exception as e:
            attempts += 1
            if is_token_limit_exceeded(e, configurable.research_model): r_msgs = remove_up_to_last_ai_message(r_msgs)
    raw_notes = "\n".join([str(m.content) for m in filter_messages(r_msgs, include_types=["tool", "ai"])])
    return {"compressed_research": "Error", "raw_notes": [raw_notes], "evidence_graph": [], "artifact_id": "error", "executive_summary": "Failed."}

researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()

def generate_argus_view(nodes: list) -> str:
    if not nodes: return "No structured evidence gathered."
    support_counts = {n.citation_index: 0 for n in nodes}
    for n in nodes:
        for s in getattr(n, 'supports', []):
            if s in support_counts: support_counts[s] += 1
    foundational = [n for n in nodes if support_counts.get(n.citation_index, 0) >= 2]
    contradicted = [n for n in nodes if getattr(n, 'contradicts', [])]
    core = [n for n in nodes if n not in foundational and n not in contradicted][:10]
    view = "### 🏛 ARGUS TOPOLOGICAL VIEW\n"
    if foundational:
        view += "**Foundational Consensus:**\n"
        for n in foundational[:5]: view += f"- [{n.citation_index}] {n.claim} (Supported by {support_counts[n.citation_index]})\n"
    if contradicted:
        view += "\n**Active Dissent:**\n"
        for n in contradicted[:5]: view += f"- [{n.citation_index}] {n.claim} (Contradicts: {n.contradicts})\n"
    if core:
        view += "\n**Peripheral Claims:**\n"
        for n in core: view += f"- [{n.citation_index}] {n.claim}\n"
    return view

async def reasoning_council(state: AgentState, config: RunnableConfig):
    tier = state.get("complexity_tier", "Medium")
    if tier in ["Simple", "Medium"]: return Command(goto="final_report_generation", update={"master_synthesis": "Standard inductive synthesis applied."})
    configurable = Configuration.from_runnable_config(config)
    raw_notes = "\n".join(state.get("notes", []))
    argus = generate_argus_view(state.get("evidence_graph", []))
    findings = argus + "\n\n### Raw Notes (Truncated)\n" + raw_notes[:8000]
    brief = state.get("research_brief", "")
    paradigms = ["Deductive", "Inductive", "Abductive", "Analogical", "Probabilistic"]
    async def run_p(p):
        mc = {"model": configurable.research_model, "max_tokens": 2048, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        try:
            res = await safe_llm_invoke(configurable_model.with_config(mc), [HumanMessage(content=reasoning_council_prompt.format(paradigm=p, brief=brief, findings=findings[:15000], master_synthesis=""))])
            return f"### {p} Perspective\n{res.content}"
        except: return f"### {p} Perspective\n[EPISTEMIC FLAG]: Skipped."
    results = await asyncio.gather(*[run_p(p) for p in paradigms])
    return Command(goto="final_report_generation", update={"master_synthesis": "\n\n".join(results)})

async def adversarial_verification(state: AgentState, config: RunnableConfig):
    tier = state.get("complexity_tier", "Medium")
    if tier in ["Simple", "Medium"]: return Command(goto="final_report_generation", update={"red_team_findings": "Skipped", "devils_advocate_critique": "Skipped", "consensus_report": state.get("master_synthesis", ""), "confidence_score": 0.8})
    configurable = Configuration.from_runnable_config(config)
    raw = state.get("master_synthesis", "")
    argus = generate_argus_view(state.get("evidence_graph", []))
    findings = argus + "\n\n### Council Synthesis\n" + raw[:5000]
    brief = state.get("research_brief", "")
    trunc = findings[:10000]
    async def run_red():
        mc = {"model": configurable.research_model, "max_tokens": 2048, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        try: return (await safe_llm_invoke(configurable_model.with_config(mc), [HumanMessage(content=red_team_prompt.format(brief=brief, findings=trunc))])).content
        except: return "[EPISTEMIC FLAG]: Red Team skipped."
    async def run_dev():
        mc = {"model": configurable.research_model, "max_tokens": 2048, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        try: return (await safe_llm_invoke(configurable_model.with_config(mc), [HumanMessage(content=devils_advocate_prompt.format(brief=brief, findings=trunc))])).content
        except: return "[EPISTEMIC FLAG]: Devil's Advocate skipped."
    red_r, dev_r = await asyncio.gather(run_red(), run_dev())
    mc = {"model": configurable.research_model, "max_tokens": 4096, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    try:
        cons_res = await safe_llm_invoke(configurable_model.with_config(mc), [HumanMessage(content=consensus_builder_prompt.format(brief=brief, findings=trunc, red_team_findings=red_r, devils_advocate_critique=dev_r))])
        cons_text = cons_res.content
        matches = re.findall(r'Confidence:\s*(0\.\d+)', cons_text)
        avg_conf = sum(float(c) for c in matches) / len(matches) if matches else 0.7
    except:
        cons_text = "[EPISTEMIC FLAG]: Consensus skipped."
        avg_conf = 0.5
    return Command(goto="final_report_generation", update={"red_team_findings": red_r, "devils_advocate_critique": dev_r, "consensus_report": cons_text, "confidence_score": avg_conf})

async def final_report_generation(state: AgentState, config: RunnableConfig):
    raw_ev = state.get("evidence_graph", [])
    v_notes = []
    if raw_ev:
        urls = [getattr(n, 'url', '') for n in raw_ev if getattr(n, 'url', '')]
        if urls:
            try:
                health = await validate_urls(urls)
                raw_ev = [n for n in raw_ev if health.get(getattr(n, 'url', ''), False)]
            except: pass
        verified = filter_and_verify_evidence(raw_ev, temporal_intent=state.get("temporal_intent", "Current"))
        for n in verified:
            d = str(n.date_published) if getattr(n, 'date_published', None) else 'Unknown'
            v_notes.append(f"Fact: {getattr(n, 'claim', '')}\nSource: {getattr(n, 'title', '')} ({getattr(n, 'url', '')})\nDate: {d}")
    notes = state.get("notes", []) + v_notes
    cleared = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    configurable = Configuration.from_runnable_config(config)
    wc = {"model": configurable.final_report_model, "max_tokens": configurable.final_report_model_max_tokens, "api_key": get_api_key_for_model(configurable.final_report_model, config), "tags": ["langsmith:nostream"]}
    retries = 0
    limit = None
    while retries <= 3:
        try:
            prompt = final_report_generation_prompt.format(research_brief=state.get("research_brief", ""), messages=get_buffer_string(state.get("messages", [])), findings=findings, date=get_today_str(), master_synthesis=state.get("master_synthesis", "Standard synthesis."), consensus_report=state.get("consensus_report", "None."), confidence_score=state.get("confidence_score", 0.8), query_paradigm=state.get("query_paradigm", "General"))
            rep = await configurable_model.with_config(wc).ainvoke([HumanMessage(content=prompt)])
            return {"final_report": rep.content, "messages": [rep], **cleared}
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                retries += 1
                if retries == 1:
                    tl = get_model_token_limit(configurable.final_report_model)
                    if not tl: return {"final_report": "Error: Token limit.", "messages": [AIMessage(content="Failed")], **cleared}
                    limit = tl * 4
                else: limit = int(limit * 0.9)
                findings = findings[:limit]
            else: return {"final_report": "Error: " + str(e), "messages": [AIMessage(content="Failed")], **cleared}
    return {"final_report": "Error: Max retries.", "messages": [AIMessage(content="Failed")], **cleared}

async def meta_learning_node(state: AgentState, config: RunnableConfig):
    conf = state.get("confidence_score", 0.8)
    iters = state.get("research_iterations", 0)
    if conf > 0.85 and iters < 4: return Command(goto=END, update={"lessons_learned": state.get("lessons_learned", [])})
    configurable = Configuration.from_runnable_config(config)
    mc = {"model": configurable.research_model, "max_tokens": 500, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    res = await configurable_model.with_config(mc).ainvoke([HumanMessage(content=meta_learning_prompt.format(confidence_score=conf, iterations=iters))])
    new_l = state.get("lessons_learned", [])
    if "LESSON:" in res.content and "Strategy optimal" not in res.content: new_l.append(res.content.strip())
    return Command(goto=END, update={"lessons_learned": new_l})

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

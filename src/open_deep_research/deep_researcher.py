"""Main LangGraph implementation for Project Omega V2."""
    existing = {node.get("node_id") for node in plan ifisinstance(node, dict)}

    targets = []
    for idx, node in enumerate((evidence_graph or [])[:3], start=1):
        claim = str(getattr(node, "claim", "")).strip()
        ifnot claim:
            continue
        targets.append({
            "node_id": f"FB_{idx}",
            "topic": f"Verify, stress-test, and resolve: {claim[:180]}",
            "depends_on": [],
        })

    ifnot targets:
        targets = [{
            "node_id": "FB_1",
            "topic": "Resolve contradictions and verify the weakest evidence nodes.",
            "depends_on": [],
        }]

    for node in targets:
        ifnode["node_id"] not in existing:
            plan.append(node)
    return plan



defgenerate_argus_view(nodes: list) -> str:
    ifnot nodes: return "No structured evidence gathered."
    support_counts = {n.citation_index: 0 for n in nodes}
    for n in nodes:
        for s in getattr(n, "supports", []):
            ifs in support_counts: support_counts[s] += 1
    foundational = [n for n in nodes ifsupport_counts.get(n.citation_index, 0) >= 2]
    contradicted = [n for n in nodes ifgetattr(n, "contradicts", [])]
    core = [n for n in nodes ifn not in foundational and n not in contradicted][:10]
    view = "### ARGUS TOPOLOGICAL VIEW\n"
    iffoundational:
        view += "Foundational Consensus:\n"
        for n in foundational[:5]: view += f"- [{n.citation_index}] {n.claim} (Supported by {support_counts[n.citation_index]})\n"
    ifcontradicted:
        view += "\nActive Dissent:\n"
        for n in contradicted[:5]: view += f"- [{n.citation_index}] {n.claim} (Contradicts: {', '.join(str(c) for c in n.contradicts)})\n"
    ifcore:
        view += "\nPeripheral Claims:\n"
        for n in core: view += f"- [{n.citation_index}] {n.claim}\n"
    return view

async defsafe_llm_invoke(model, messages):
    async with groq_burst_semaphore:
        for attempt in range(3):
            try: return await model.ainvoke(messages)
            except Exception as e:
                err = str(e).lower()
                if"rate limit" in err or "429" in err or "resource_exhausted" in err or "timeout" in err:
                    new_key = _shield.get_key()
                    model = model.with_config({"api_key": new_key})
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                ifattempt == 2: raise RuntimeError("[EPISTEMIC FLAG]: LLM infrastructure constraint.")
        raise RuntimeError("[EPISTEMIC FLAG]: LLM infrastructure constraint.")

async defclarify_with_user(state: AgentState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        ifnot configurable.allow_clarification: return Command(goto="write_research_brief")
        mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        cm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(ClarifyWithUser)
        prompt = clarify_with_user_instructions.format(messages=get_buffer_string(state["messages"]), date=get_today_str())
        response = await safe_llm_invoke(cm, [HumanMessage(content=prompt)])
        ifresponse.need_clarification: return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})
    except Exception as e:
        logging.error(f"clarify_with_user failed: {e}")
        return Command(goto="write_research_brief")

async defwrite_research_brief(state: AgentState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(ResearchQuestion)
        prompt = transform_messages_into_research_topic_prompt.format(messages=get_buffer_string(state.get("messages", [])), date=get_today_str())
        response = await safe_llm_invoke(rm, [HumanMessage(content=prompt)])
        mem_ctx = omega_memory.get_context_prompt()
        sup_sys = lead_researcher_prompt.format(date=get_today_str(), mcp_prompt=configurable.mcp_prompt or "", max_concurrent_research_units=configurable.max_concurrent_research_units, max_researcher_iterations=configurable.max_researcher_iterations, temporal_intent=getattr(response, "temporal_intent", "Current"), complexity_tier="Pending", lessons_learned=mem_ctx)
        return Command(goto="meta_cognitive_router", update={"research_brief": response.research_brief, "temporal_intent": getattr(response, "temporal_intent", "Current"), "hard_constraints": getattr(response, "hard_constraints", []), "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=response.research_brief)]}})
    except Exception as e:
        logging.error(f"write_research_brieffailed: {e}")
        return Command(goto=END)

async defmeta_cognitive_router(state: AgentState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        mc = {"model": configurable.research_model, "max_tokens": 4096, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(RouterDecision)
        mem_ctx = omega_memory.get_context_prompt()
        prompt = meta_cognitive_router_prompt.format(research_brief=state.get("research_brief", ""), date=get_today_str(), memory_context=mem_ctx)
        response = await safe_llm_invoke(rm, [HumanMessage(content=prompt)])
        sup_sys = lead_researcher_prompt.format(date=get_today_str(), mcp_prompt=configurable.mcp_prompt or "", max_concurrent_research_units=response.dynamic_research_units, max_researcher_iterations=response.dynamic_tool_budget, complexity_tier=response.complexity_tier, temporal_intent=state.get("temporal_intent", "Current"), lessons_learned=mem_ctx)
        plan_dicts = [n.model_dump() for n in response.research_plan] ifresponse.research_plan else []
        return Command(goto="research_supervisor", update={"query_paradigm": response.query_paradigm, "complexity_tier": response.complexity_tier, "dynamic_tool_budget": response.dynamic_tool_budget, "dynamic_research_units": response.dynamic_research_units, "research_plan": plan_dicts, "completed_nodes": [], "supervisor_messages": {"type": "override", "value": [SystemMessage(content=sup_sys), HumanMessage(content=state.get("research_brief", ""))]}})
    except Exception as e:
        logging.error(f"meta_cognitive_router failed: {e}")
        return Command(goto=END)
async defsupervisor(state: SupervisorState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).bind_tools([ConductResearch, ResearchComplete, think_tool])
        saturation = calculate_epistemic_saturation(state.get("evidence_graph", []), state.get("research_plan", []))
        ifsaturation >= 0.85:
            return Command(goto="supervisor_tools", update={"supervisor_messages": [AIMessage(content="", tool_calls=[{"name": "ResearchComplete", "args": {}, "id": "forced_halt"}])], "research_iterations": state.get("research_iterations", 0) + 1})
        sup_msgs = list(state.get("supervisor_messages", []))
        core_msgs = [m for m in sup_msgs ifisinstance(m, (SystemMessage, HumanMessage)) and "DAG_STATUS" not in str(getattr(m, "content", ""))]
        recent_msgs = [m for m in sup_msgs ifnot isinstance(m, (SystemMessage, HumanMessage))][-4:]
        sup_msgs = core_msgs + recent_msgs
        plan = state.get("research_plan", [])
        completed = state.get("completed_nodes", [])
        ifplan:
            dag_status = f"
<DAG_STATUS>\nPlan: {plan}\nCompleted Nodes: {completed}\n</DAG_STATUS>"
            sup_msgs.append(SystemMessage(content=dag_status))
        response = await safe_llm_invoke(rm, sup_msgs)
        return Command(goto="supervisor_tools", update={"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1})
    except Exception as e:
        logging.error(f"supervisor failed: {e}")
        return Command(goto=END)

async defsupervisor_tools(state: SupervisorState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        sup_msgs = state.get("supervisor_messages", [])
        ifnot sup_msgs: return Command(goto=END)
        iters = state.get("research_iterations", 0)
        last_msg = sup_msgs[-1]
        ifiters > configurable.max_researcher_iterations or not getattr(last_msg, "tool_calls", None) or any(tc["name"] == "ResearchComplete" for tc in last_msg.tool_calls):
            return Command(goto=END, update={"notes": get_notes_from_tool_calls(sup_msgs), "research_brief": state.get("research_brief", "")})
        all_tool_msgs = []
        update_payload = {"supervisor_messages": []}
        conduct_calls = [t for t in last_msg.tool_calls ift["name"] == "ConductResearch"]
        ifconduct_calls:
            allowed = conduct_calls[:configurable.max_concurrent_research_units]
            tasks = []
            for tc in allowed:
                base_topic = tc["args"]["research_topic"]
                invariant_payload = f"{base_topic}\n\n[INVARIANT CONSTRAINTS]\nTemporal: {state.get('temporal_intent')}\nHard Constraints: {state.get('hard_constraints')}"
                tasks.append(researcher_subgraph.ainvoke({"researcher_messages": [HumanMessage(content=invariant_payload)], "research_topic": invariant_payload}, config))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            vfs_update, agg_graph = {}, []
            for obs, tc in zip(results, allowed):
                ifisinstance(obs, Exception):
                    all_tool_msgs.append(ToolMessage(content=f"[FALLBACK] Researcher failed: {str(obs)}", name=tc["name"], tool_call_id=tc["id"]))
                    continue
                art_id = str(obs.get("artifact_id", tc["id"]))
                vfs_update[art_id] = obs.get("compressed_research", "")
                all_tool_msgs.append(ToolMessage(content=f"ARTIFACT ID: {art_id}\nSUMMARY: {obs.get('executive_summary', 'Done')}\n[VFS STORED]", name=tc["name"], tool_call_id=tc["id"]))
                agg_graph.extend(obs.get("evidence_graph", []))
            ifvfs_update: update_payload["virtual_filesystem"] = vfs_update
            ifagg_graph: update_payload["evidence_graph"] = agg_graph
            newly_completed = [tc["args"].get("node_id", "") for tc in allowed iftc["args"].get("node_id")]
            ifnewly_completed: update_payload["completed_nodes"] = list(set(state.get("completed_nodes", [])).union(set(newly_completed)))
        update_payload["supervisor_messages"] = all_tool_msgs
        return Command(goto="supervisor", update=update_payload)
    except Exception as e:
        logging.error(f"supervisor_tools failed: {e}")
        return Command(goto=END)

supervisor_builder = AsyncStateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder = AsyncStateGraph(SupervisorState, output_schema=ResearcherOutputState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()

async defresearcher(state: ResearcherState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        tools = await get_all_tools(config)
        mc = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        prompt = research_system_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_str(), temporal_intent=state.get("temporal_intent", "Current"))
        rm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).bind_tools(tools)
        r_msgs = state.get("researcher_messages", [])
        core_r = [m for m in r_msgs ifisinstance(m, (SystemMessage, HumanMessage))]
        recent_r = [m for m in r_msgs ifnot isinstance(m, (SystemMessage, HumanMessage))][-6:]
        msgs = [SystemMessage(content=prompt)] + core_r + recent_r
        response = await safe_llm_invoke(rm, msgs)
        return Command(goto="researcher_tools", update={"researcher_messages": [response], "tool_call_iterations": state.get("tool_call_iterations", 0) + 1})
    except Exception as e:
        logging.error(f"researcher failed: {e}")
        return Command(goto="compress_research")

async defexecute_tool_safely(tool, args, config):
    for attempt in range(3):
        try: return await tool.ainvoke(args, config)
        except Exception as e:
            if"rate limit" in str(e).lower() or "429" in str(e).lower(): await asyncio.sleep(2 ** attempt)
            else: return f"[TOOL FALLBACK]: {getattr(tool, 'name', 'unknown')} failed."
    return "[TOOL FALLBACK]: Max retries exceeded."

async defresearcher_tools(state: ResearcherState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        r_msgs = state.get("researcher_messages", [])
        ifnot r_msgs: return Command(goto="compress_research")
        last_msg = r_msgs[-1]
        ifnot getattr(last_msg, "tool_calls", None): return Command(goto="compress_research")
        tools = await get_all_tools(config)
        tools_by_name = {t.name: t for t in tools ifhasattr(t, "name")}
        obs = await asyncio.gather(*[execute_tool_safely(tools_by_name[tc["name"]], tc["args"], config) for tc in last_msg.tool_calls iftc["name"] in tools_by_name])
        tool_outputs = [ToolMessage(content=o, name=tc["name"], tool_call_id=tc["id"]) for o, tc in zip(obs, last_msg.tool_calls) iftc["name"] in tools_by_name]
        new_claims = [o for o in obs ifisinstance(o, str)]
        existing_context = [m.content for m in r_msgs ifhasattr(m, "content") and isinstance(m.content, str)]
        ifcheck_information_satiation(new_claims, existing_context) or state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls:
            return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
        return Command(goto="researcher", update={"researcher_messages": tool_outputs})
    except Exception as e:
        logging.error(f"researcher_tools failed: {e}")
        return Command(goto="compress_research")

async defcompress_research(state: ResearcherState, config: RunnableConfig):
    try:
        configurable = Configuration.from_runnable_config(config)
        mc = {"model": configurable.compression_model, "max_tokens": configurable.compression_model_max_tokens, "api_key": get_api_key_for_model(configurable.compression_model, config), "tags": ["langsmith:nostream"]}
        sm = init_chat_model(model=mc["model"], max_tokens=mc["max_tokens"], api_key=mc["api_key"]).with_structured_output(EvidenceGraphExtraction)
        r_msgs = state.get("researcher_messages", []) + [HumanMessage(content=compress_research_simple_human_message)]
        msgs = [SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))] + r_msgs
        response = await sm.ainvoke(msgs)
        raw_notes = "\n".join([str(m.content) for m in filter_messages(r_msgs, include_types=["tool", "ai"])])
        art_id = hashlib.md5(raw_notes.encode()).hexdigest()[:8]
        readable = "Extracted Evidence Graph:\n" + "\n".join([f"Fact {i+1}: {n.claim} ({n.url})" for i, n in enumerate(response.nodes)])
        for node in response.nodes:
            omega_local_memory.store(getattr(node, 'claim', ''), getattr(node, 'url', ''))
        return {"compressed_research": readable, "raw_notes": [raw_notes], "evidence_graph": response.nodes, "artifact_id": art_id, "executive_summary": readable[:500]}
    except Exception as e:
        logging.error(f"compress_research failed: {e}")
        return {"compressed_research": "Error", "raw_notes": [], "evidence_graph": [], "artifact_id": "error", "executive_summary": "Failed"}

researcher_builder = AsyncStateGraph(AsyncAsyncStateGraph(ResearcherState, output_schema=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()
async defreasoning_council(state: AgentState, config: RunnableConfig):
    try:
        tier = state.get("complexity_tier", "Medium")
        iftier in ["Simple", "Medium"]:
            return Command(goto="adversarial_verification", update={"master_synthesis": "Standard inductive synthesis applied."})
        configurable = Configuration.from_runnable_config(config)
        raw_notes = "\n".join(state.get("notes", []))
        argus = generate_argus_view(state.get("evidence_graph", []))
        findings = argus + "\n\n### Raw Notes\n" + raw_notes[:8000]
        brief= state.get("research_brief", "")
        paradigms = ["Deductive", "Inductive", "Abductive", "Analogical", "Probabilistic"]
        async defrun_p(p):
            mc = {"model": configurable.research_model, "max_tokens": 2048, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
            try:
                prompt = reasoning_council_prompt.format(paradigm=p, brief=brief, findings=findings[:15000], master_synthesis="")
                res = await safe_llm_invoke(configurable_model.with_config(mc), [HumanMessage(content=prompt)])
                return f"### {p} Perspective\n{res.content}"
            except Exception: return f"### {p} Perspective\n[EPISTEMIC FLAG]: Skipped."
        results = await asyncio.gather(*[run_p(p) for p in paradigms])
        return Command(goto="adversarial_verification", update={"master_synthesis": "\n\n".join(results)})
    except Exception as e:
        logging.error(f"reasoning_council failed: {e}")
        return Command(goto="adversarial_verification", update={"master_synthesis": "Council failed."})

async defadversarial_verification(state: AgentState, config: RunnableConfig):
    try:
        evidence = state.get("evidence_graph", [])
        temporal = state.get("temporal_intent", "Current")
        verification_results = programmatic_epistemic_verification(evidence, temporal)
        argus = generate_argus_view(evidence) ifevidence else ""
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

async deffinal_report_generation(state: AgentState, config: RunnableConfig):
    confidence_score = float(state.get("confidence_score", 0.0) or 0.0)
    contradiction_count = sum(1 for n in state.get("evidence_graph", []) ifgetattr(n, "contradicts", []))
    ifconfidence_score < 0.65 or contradiction_count > 0:
        return Command(
            goto="supervisor",
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
        ifraw_ev:
            urls = [getattr(n, "url", "") for n in raw_ev ifgetattr(n, "url", "")]
            ifurls:
                try:
                    health = await validate_urls(urls)
                    raw_ev = [n for n in raw_ev ifhealth.get(getattr(n, "url", ""), False)]
                except Exception: pass
        verified = filter_and_verify_evidence(raw_ev, temporal_intent=state.get("temporal_intent", "Current"))
        verified = await verify_citations_programmatically(verified)
        for n in verified:
            d = str(n.date_published) ifgetattr(n, "date_published", None) else "Unknown"
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
                    query_paradigm=state.get("query_paradigm", "General"),
                )
                rep = await configurable_model.with_config(wc).ainvoke([HumanMessage(content=prompt)])
                return {"final_report": rep.content, "messages": [rep], **cleared}
            except Exception as e:
                ifis_token_limit_exceeded(e, configurable.final_report_model):
                    retries += 1
                    ifretries == 1:
                        tl = get_model_token_limit(configurable.final_report_model)
                        ifnot tl: return {"final_report": "Error: Token limit.", "messages": [AIMessage(content="Failed")], **cleared}
                        limit = tl * 4
                    else: limit = int(limit * 0.9)
                    findings = findings[:limit]
                else: return {"final_report": f"Error: {str(e)}", "messages": [AIMessage(content="Failed")], **cleared}
        return {"final_report": "Error: Max retries.", "messages": [AIMessage(content="Failed")], **cleared}
    except Exception as e:
        logging.error(f"final_report_generation failed: {e}")
        return {"final_report": f"Fatal Error: {str(e)}", "messages": [AIMessage(content="Failed")]}

async defmeta_learning_node(state: AgentState, config: RunnableConfig):
    try:
        conf= state.get("confidence_score", 0.8)
        iters = state.get("research_iterations", 0)
        ifconf> 0.85 and iters < 4: return Command(goto=END, update={"lessons_learned": state.get("lessons_learned", [])})
        configurable = Configuration.from_runnable_config(config)
        mc = {"model": configurable.research_model, "max_tokens": 500, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        res = await safe_llm_invoke(configurable_model.with_config(mc), [HumanMessage(content=meta_learning_prompt.format(confidence_score=conf, iterations=iters))])
        new_l = state.get("lessons_learned", [])
        if"LESSON:" in res.content and "Strategy optimal" not in res.content: new_l.append(res.content.strip())
        if"github.com" in str(state.get("notes", [])): omega_memory.update_domain("github.com", True)
        return Command(goto=END, update={"lessons_learned": new_l})
    except Exception: return Command(goto=END)

builder = AsyncStateGraph(AgentState, input_schema=AgentInputState, config_schema=Configuration)
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
"""Main LangGraph implementation for Project Omega."""
import asyncio
import logging
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
    meta_learning_prompt,
    reasoning_council_prompt,
    red_team_prompt,
    devils_advocate_prompt,
    consensus_builder_prompt,
    meta_cognitive_router_prompt,
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
    remove_up_to_last_ai_message,
    validate_urls, think_tool,
)

configurable_model = init_chat_model(configurable_fields=("model", "max_tokens", "api_key"))

async def clarify_with_user(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")
    messages = state["messages"]
    model_config = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    clarification_model = configurable_model.with_structured_output(ClarifyWithUser).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(model_config)
    prompt_content = clarify_with_user_instructions.format(messages=get_buffer_string(messages), date=get_today_str())
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})

async def write_research_brief(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    research_model = configurable_model.with_structured_output(ResearchQuestion).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    prompt_content = transform_messages_into_research_topic_prompt.format(messages=get_buffer_string(state.get("messages", [])), date=get_today_str())
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        mcp_prompt=configurable.mcp_prompt or "",
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
        temporal_intent=getattr(response, "temporal_intent", "Current"),
        complexity_tier="Pending",
        lessons_learned="
".join(state.get("lessons_learned", []))
    )
    return Command(goto="meta_cognitive_router", 
        update={
            "research_brief": response.research_brief,
        "temporal_intent": getattr(response, "temporal_intent", "Current"),
        "hard_constraints": getattr(response, "hard_constraints", []),
        "temporal_intent": getattr(response, "temporal_intent", "Current"),
            "supervisor_messages": {"type": "override", "value": [SystemMessage(content=supervisor_system_prompt), HumanMessage(content=response.research_brief)]}})


async def meta_cognitive_router(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    model_config = {"model": configurable.research_model, "max_tokens": 4096, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    router_model = configurable_model.with_structured_output(RouterDecision).with_config(model_config)
    prompt_content = meta_cognitive_router_prompt.format(research_brief=state.get("research_brief", ""), date=get_today_str())
    response = await router_model.ainvoke([HumanMessage(content=prompt_content)])
    
    supervisor_sys = lead_researcher_prompt.format(
        date=get_today_str(),
        mcp_prompt=configurable.mcp_prompt or "",
        max_concurrent_research_units=response.dynamic_research_units,
        max_researcher_iterations=response.dynamic_tool_budget,
        complexity_tier=response.complexity_tier,
        temporal_intent=state.get("temporal_intent", "Current"),
        lessons_learned="\n".join(state.get("lessons_learned", []))
    ), max_concurrent_research_units=response.dynamic_research_units, max_researcher_iterations=response.dynamic_tool_budget, complexity_tier=response.complexity_tier)
    return Command(goto="research_supervisor", update={
        "query_paradigm": response.query_paradigm,
        "complexity_tier": response.complexity_tier,
        "dynamic_tool_budget": response.dynamic_tool_budget,
        "dynamic_research_units": response.dynamic_research_units,
        "supervisor_messages": {"type": "override", "value": [SystemMessage(content=supervisor_sys), HumanMessage(content=state.get("research_brief", ""))]}
    })

async def supervisor(state: SupervisorState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    research_model = configurable_model.bind_tools(lead_researcher_tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    return Command(goto="supervisor_tools", update={"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1})

async def supervisor_tools(state: SupervisorState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    if research_iterations > configurable.max_researcher_iterations or not most_recent_message.tool_calls or any(tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls):
        return Command(goto=END, update={"notes": get_notes_from_tool_calls(supervisor_messages), "research_brief": state.get("research_brief", "")})
    
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    for tc in [t for t in most_recent_message.tool_calls if t["name"] == "think_tool"]:
        all_tool_messages.append(ToolMessage(content="Reflection recorded: " + tc["args"]["reflection"], name="think_tool", tool_call_id=tc["id"]))
        
    conduct_calls = [t for t in most_recent_message.tool_calls if t["name"] == "ConductResearch"]
    if conduct_calls:
        allowed = conduct_calls[:configurable.max_concurrent_research_units]
        overflow = conduct_calls[configurable.max_concurrent_research_units:]
        try:
            tasks = [researcher_subgraph.ainvoke({"researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])], "research_topic": tc["args"]["research_topic"]}, config) for tc in allowed]
            results = await asyncio.gather(*tasks)
            for obs, tc in zip(results, allowed):
                all_tool_messages.append(ToolMessage(content=obs.get("compressed_research", "Error"), name=tc["name"], tool_call_id=tc["id"]))
            for tc in overflow:
                all_tool_messages.append(ToolMessage(content="Error: Exceeded max concurrent units.", name="ConductResearch", tool_call_id=tc["id"]))
            raw_concat = "\n".join(["\n".join(obs.get("raw_notes", [])) for obs in results])
            if raw_concat: update_payload["raw_notes"] = [raw_concat]
        except Exception as e:
            return Command(goto=END, update={"notes": get_notes_from_tool_calls(supervisor_messages), "research_brief": state.get("research_brief", "")})
            
    update_payload["supervisor_messages"] = all_tool_messages
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
    research_model_config = {"model": configurable.research_model, "max_tokens": configurable.research_model_max_tokens, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str(),
        temporal_intent=state.get("temporal_intent", "Current")
    ), temporal_intent=state.get("temporal_intent", "Current"))
    research_model = configurable_model.bind_tools(tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    messages = [SystemMessage(content=researcher_prompt)] + state.get("researcher_messages", [])
    response = await research_model.ainvoke(messages)
    return Command(goto="researcher_tools", update={"researcher_messages": [response], "tool_call_iterations": state.get("tool_call_iterations", 0) + 1})

async def execute_tool_safely(tool, args, config):
    try: return await tool.ainvoke(args, config)
    except Exception as e: return "Error executing tool: " + str(e)

async def researcher_tools(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    if not most_recent_message.tool_calls and not (openai_websearch_called(most_recent_message) or anthropic_websearch_called(most_recent_message)):
        return Command(goto="compress_research")
        
    tools = await get_all_tools(config)
    tools_by_name = {t.name if hasattr(t, "name") else t.get("name", "web_search"): t for t in tools}
    observations = await asyncio.gather(*[execute_tool_safely(tools_by_name[tc["name"]], tc["args"], config) for tc in most_recent_message.tool_calls])
    tool_outputs = [ToolMessage(content=obs, name=tc["name"], tool_call_id=tc["id"]) for obs, tc in zip(observations, most_recent_message.tool_calls)]
    
    new_claims = [obs for obs in observations if isinstance(obs, str)]
    existing_context = [m.content for m in researcher_messages if hasattr(m, 'content') and isinstance(m.content, str)]
    if check_information_satiation(new_claims, existing_context):
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
        
    if state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls or any(tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls):
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
        
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})

async def compress_research(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({"model": configurable.compression_model, "max_tokens": configurable.compression_model_max_tokens, "api_key": get_api_key_for_model(configurable.compression_model, config), "tags": ["langsmith:nostream"]})
    researcher_messages = state.get("researcher_messages", [])
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    synthesis_attempts = 0
    while synthesis_attempts < 3:
        try:
            messages = [SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))] + researcher_messages
            structured_model = synthesizer_model.with_structured_output(EvidenceGraphExtraction)
            response = await structured_model.ainvoke(messages)
            raw_notes_content = "\n".join([str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])])
            
            readable_text = "Extracted Evidence Graph:\n"
            for i, node in enumerate(response.nodes):
                readable_text += "Fact " + str(i+1) + ": " + str(getattr(node, 'claim', '')) + "\nSource: " + str(getattr(node, 'title', getattr(node, 'source_title', ''))) + " (" + str(getattr(node, 'url', getattr(node, 'source_url', ''))) + ")\n"
                
            return {"compressed_research": readable_text, "raw_notes": [raw_notes_content], "evidence_graph": response.nodes}
        except Exception as e:
            synthesis_attempts += 1
            if is_token_limit_exceeded(e, configurable.research_model): researcher_messages = remove_up_to_last_ai_message(researcher_messages)
            
    raw_notes_content = "\n".join([str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])])
    return {"compressed_research": "Error synthesizing research report.", "raw_notes": [raw_notes_content], "evidence_graph": []}

researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def reasoning_council(state: AgentState, config: RunnableConfig):
    tier = state.get("complexity_tier", "Medium")
    # Intelligent Adaptation: Skip heavy reasoning for simple queries
    if tier in ["Simple", "Medium"]:
        return Command(goto="final_report_generation", update={"master_synthesis": "Standard inductive synthesis applied."})
        
    configurable = Configuration.from_runnable_config(config)
    findings = "\n".join(state.get("notes", []))
    brief = state.get("research_brief", "")
    paradigms = ["Deductive", "Inductive", "Abductive", "Analogical", "Probabilistic"]
    
    async def run_paradigm(paradigm):
        # Truncate findings to 15k chars to protect Groq context limits during parallel calls
        prompt = reasoning_council_prompt.format(paradigm=paradigm, brief=brief, findings=findings[:15000], master_synthesis="")
        model_config = {"model": configurable.research_model, "max_tokens": 2048, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        try:
            res = await configurable_model.with_config(model_config).ainvoke([HumanMessage(content=prompt)])
            return f"### {paradigm} Perspective\n{res.content}"
        except Exception as e:
            return f"### {paradigm} Perspective\nFailed: {str(e)}"
            
    import asyncio
    # Spawn all 5 paradigms in parallel using the Groq Multi-Key Pool
    tasks = [run_paradigm(p) for p in paradigms]
    results = await asyncio.gather(*tasks)
    synthesis = "\n\n".join(results)
    
    return Command(goto="final_report_generation", update={"master_synthesis": synthesis})


async def adversarial_verification(state: AgentState, config: RunnableConfig):
    tier = state.get("complexity_tier", "Medium")
    # Only run adversarial verification for Complex and Expert tiers
    if tier in ["Simple", "Medium"]:
        return Command(goto="final_report_generation", update={
            "red_team_findings": "Skipped (Tier: " + tier + ")",
            "devils_advocate_critique": "Skipped (Tier: " + tier + ")",
            "consensus_report": state.get("master_synthesis", "No reasoning council findings."),
            "confidence_score": 0.8
        })
    
    configurable = Configuration.from_runnable_config(config)
    findings = state.get("master_synthesis", "")
    brief = state.get("research_brief", "")
    
    # Truncate to protect Groq context during parallel calls
    findings_truncated = findings[:10000] if len(findings) > 10000 else findings
    
    async def run_red_team():
        prompt = red_team_prompt.format(brief=brief, findings=findings_truncated)
        model_config = {"model": configurable.research_model, "max_tokens": 2048, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        try:
            res = await configurable_model.with_config(model_config).ainvoke([HumanMessage(content=prompt)])
            return res.content
        except Exception as e:
            return "Red Team failed: " + str(e)
    
    async def run_devils_advocate():
        prompt = devils_advocate_prompt.format(brief=brief, findings=findings_truncated)
        model_config = {"model": configurable.research_model, "max_tokens": 2048, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
        try:
            res = await configurable_model.with_config(model_config).ainvoke([HumanMessage(content=prompt)])
            return res.content
        except Exception as e:
            return "Devil's Advocate failed: " + str(e)
    
    # Run Red Team and Devil's Advocate in parallel
    import asyncio
    red_team_result, devils_result = await asyncio.gather(run_red_team(), run_devils_advocate())
    
    # Now run Consensus Builder with all inputs
    consensus_prompt = consensus_builder_prompt.format(
        brief=brief,
        findings=findings_truncated,
        red_team_findings=red_team_result,
        devils_advocate_critique=devils_result
    )
    model_config = {"model": configurable.research_model, "max_tokens": 4096, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    
    try:
        consensus_res = await configurable_model.with_config(model_config).ainvoke([HumanMessage(content=consensus_prompt)])
        consensus_text = consensus_res.content
        
        # Extract average confidence score from the consensus text
        import re
        confidence_matches = re.findall(r'Confidence:\s*(0\.\d+)', consensus_text)
        if confidence_matches:
            avg_confidence = sum(float(c) for c in confidence_matches) / len(confidence_matches)
        else:
            avg_confidence = 0.7
            
    except Exception as e:
        consensus_text = "Consensus building failed: " + str(e)
        avg_confidence = 0.5
    
    return Command(goto="final_report_generation", update={
        "red_team_findings": red_team_result,
        "devils_advocate_critique": devils_result,
        "consensus_report": consensus_text,
        "confidence_score": avg_confidence
    })

async def final_report_generation(state: AgentState, config: RunnableConfig):
    raw_evidence = state.get("evidence_graph", [])
    verified_notes = []
    if raw_evidence:
        
        # --- POST-HOC URL VALIDATION (Flaw 6 Fix) ---
        urls_to_check = [getattr(n, 'url', '') for n in raw_evidence if getattr(n, 'url', '')]
        if urls_to_check:
            try:
                url_health = await validate_urls(urls_to_check)
                raw_evidence = [n for n in raw_evidence if url_health.get(getattr(n, 'url', ''), False)]
            except Exception:
                pass # Fallback to unvalidated if network fails
        # ---------------------------------------------

        verified = filter_and_verify_evidence(raw_evidence, temporal_intent=state.get("temporal_intent", "Current"))
        for node in verified:
            date_str = str(node.date_published) if getattr(node, 'date_published', None) else 'Unknown'
            verified_notes.append("Fact: " + str(getattr(node, 'claim', '')) + "\nSource: " + str(getattr(node, 'title', '')) + " (" + str(getattr(node, 'url', '')) + ")\nDate: " + date_str)
            
    notes = state.get("notes", []) + verified_notes
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {"model": configurable.final_report_model, "max_tokens": configurable.final_report_model_max_tokens, "api_key": get_api_key_for_model(configurable.final_report_model, config), "tags": ["langsmith:nostream"]}
    
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    while current_retry <= max_retries:
        try:
            final_report_prompt = final_report_generation_prompt.format(
            research_brief=state.get("research_brief", ""),
            messages=get_buffer_string(state.get("messages", [])),
            findings=findings,
            date=get_today_str(),
            master_synthesis=state.get("master_synthesis", "Standard synthesis applied."),
            consensus_report=state.get("consensus_report", "No adversarial verification performed."),
            confidence_score=state.get("confidence_score", 0.8),
            query_paradigm=state.get("query_paradigm", "General")
        )
        final_report = await configurable_model.with_config(writer_model_config).ainvoke([HumanMessage(content=final_report_prompt)])
            return {"final_report": final_report.content, "messages": [final_report], **cleared_state}
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                if current_retry == 1:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit: return {"final_report": "Error: Token limit exceeded.", "messages": [AIMessage(content="Failed")], **cleared_state}
                    findings_token_limit = model_token_limit * 4
                else: findings_token_limit = int(findings_token_limit * 0.9)
                findings = findings[:findings_token_limit]
            else: return {"final_report": "Error generating final report: " + str(e), "messages": [AIMessage(content="Failed")], **cleared_state}
    return {"final_report": "Error: Max retries exceeded.", "messages": [AIMessage(content="Failed")], **cleared_state}


async def meta_learning_node(state: AgentState, config: RunnableConfig):
    conf = state.get("confidence_score", 0.8)
    iters = state.get("research_iterations", 0)
    if conf > 0.85 and iters < 4:
        return Command(goto=END, update={"lessons_learned": state.get("lessons_learned", [])})
    
    configurable = Configuration.from_runnable_config(config)
    model_config = {"model": configurable.research_model, "max_tokens": 500, "api_key": get_api_key_for_model(configurable.research_model, config), "tags": ["langsmith:nostream"]}
    prompt = meta_learning_prompt.format(confidence_score=conf, iterations=iters)
    res = await configurable_model.with_config(model_config).ainvoke([HumanMessage(content=prompt)])
    
    new_lessons = state.get("lessons_learned", [])
    if "LESSON:" in res.content and "Strategy optimal" not in res.content:
        new_lessons.append(res.content.strip())
    return Command(goto=END, update={"lessons_learned": new_lessons})

deep_researcher_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("reasoning_council", reasoning_council)
    deep_researcher_builder.add_node("adversarial_verification", adversarial_verification)
    deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_node("meta_learning_node", meta_learning_node)
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "reasoning_council")
    deep_researcher_builder.add_edge("reasoning_council", "adversarial_verification")
    deep_researcher_builder.add_edge("adversarial_verification", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", "meta_learning_node")
deep_researcher_builder.add_edge("meta_learning_node", END)

memory = MemorySaver()
deep_researcher = deep_researcher_builder.compile(checkpointer=memory)
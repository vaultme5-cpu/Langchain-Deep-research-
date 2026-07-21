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
)
from open_deep_research.state import (
    AgentInputState, AgentState, ClarifyWithUser, ConductResearch,
    EvidenceGraphExtraction, ResearchComplete, ResearcherOutputState,
    ResearcherState, ResearchQuestion, SupervisorState,
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
    supervisor_system_prompt = lead_researcher_prompt.format(date=get_today_str(), max_concurrent_research_units=configurable.max_concurrent_research_units, max_researcher_iterations=configurable.max_researcher_iterations)
    return Command(goto="research_supervisor", update={"research_brief": response.research_brief, "supervisor_messages": {"type": "override", "value": [SystemMessage(content=supervisor_system_prompt), HumanMessage(content=response.research_brief)]}})

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
    researcher_prompt = research_system_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_str())
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

async def final_report_generation(state: AgentState, config: RunnableConfig):
    raw_evidence = state.get("evidence_graph", [])
    verified_notes = []
    if raw_evidence:
        verified = filter_and_verify_evidence(raw_evidence)
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
            final_report_prompt = final_report_generation_prompt.format(research_brief=state.get("research_brief", ""), messages=get_buffer_string(state.get("messages", [])), findings=findings, date=get_today_str())
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

deep_researcher_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

memory = MemorySaver()
deep_researcher = deep_researcher_builder.compile(checkpointer=memory)
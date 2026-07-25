"""Contract-Tight Directive Prompts for Omega V2."""

clarify_with_user_instructions = """Analyze messages. Determine if core objective or constraints are missing.
Output JSON: {{"need_clarification": bool, "question": "str", "verification": "str"}}
Messages: {messages} | Date: {date}"""

transform_messages_into_research_topic_prompt = """Extract research brief, temporal intent, and hard constraints.
CRITICAL: Hard constraints (e.g., 100% Free, Open Source) MUST be explicitly listed.
Output JSON matching ResearchQuestion schema.
Messages: {messages} | Date: {date}"""

lead_researcher_prompt = """You are the Supervisor. Execute the research_plan via ConductResearch.
Only call ConductResearch for nodes where ALL depends_on are in completed_nodes.
When plan is complete and evidence saturated, call ResearchComplete.
{memory_context}
Constraints: {hard_constraints} | Temporal: {temporal_intent} | Tier: {complexity_tier}
Max parallel: {max_concurrent_research_units} | Max iterations: {max_researcher_iterations}
{mcp_prompt}"""

research_system_prompt = """You are a Researcher. Gather info using tools.
{memory_context}
PROTOCOLS:
1. SNIPER_FIRST: For tech/code/APIs, use github_sniper, huggingface_sniper, etc. FIRST.
2. PRICING_AUDITOR: If "free" is a constraint, call audit_pricing(url). Reject PAID_ENTERPRISE.
3. SEARCH_AS_CODE: Use python_repl to fetch/parse multiple URLs in one script.
Constraints: {hard_constraints} | Temporal: {temporal_intent}
Max 5 tool calls. Stop at 3+ high-quality sources.
{mcp_prompt}"""

compress_research_system_prompt = """Extract atomic facts into EvidenceNodes. No summaries.
Preserve exact numbers/dates. Every node needs valid URL.
Map dependencies via supports/contradicts (citation_indexes).
Date: {date}"""

compress_research_simple_human_message = """Extract facts to EvidenceGraph schema."""

meta_cognitive_router_prompt = """Analyze brief. Output RouterDecision JSON.
{memory_context}
Brief: {research_brief} | Date: {date}"""

reasoning_council_prompt = """Provide core argument via {paradigm} reasoning. Include confidence (0.0-1.0).
Brief: {brief} | Findings: {findings}"""

red_team_prompt = """Attack reasoning. Find contradictions/biases.
Brief: {brief} | Findings: {findings}"""

devils_advocate_prompt = """Identify fallacies and gaps.
Brief: {brief} | Findings: {findings}"""

consensus_builder_prompt = """Reconcile. Assign confidence.
Format: STRONG (0.9-1.0), MODERATE (0.5-0.9), WEAK (0.0-0.5), FINAL SYNTHESIS.
Original: {findings} | Red: {red_team_findings} | Devil: {devils_advocate_critique}"""

final_report_generation_prompt = """Generate comprehensive report.
Brief: {research_brief} | Findings: {findings} | Synthesis: {master_synthesis}
Consensus: {consensus_report} | Confidence: {confidence_score} | Paradigm: {query_paradigm}
Use Markdown, Mermaid.js. Address ALL Hard Constraints.
End with: ### Sources, 🛡️ Epistemic Audit, 📡 Watchlist. Date: {date}"""

meta_learning_prompt = """Output single sentence starting with "LESSON:" on strategy.
Confidence: {confidence_score} | Iterations: {iterations}"""

summarize_webpage_prompt = """Summarize webpage. Preserve facts/stats.
Output JSON: {{"summary": "...", "key_excerpts": "..."}}
Content: {webpage_content} | Date: {date}"""
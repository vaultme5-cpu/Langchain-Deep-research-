"""Contract-Tight Directive Prompts for Omega V2."""

clarify_with_user_instructions = """Analyze messages. Determine if core objective or constraints are missing.
Output JSON: { "need_clarification": bool, "question": "str", "verification": "str" }
Messages: {messages}| Date: {date}"""

transform_messages_into_research_topic_prompt = """Extract research brief, temporal intent, and hard constraints.
CRITICAL: Hard constraints (e.g., 100% Free, Open Source) MUST be explicitly listed.
Output JSON matching ResearchQuestion schema.
Messages: {messages}| Date: {date}"""

lead_researcher_prompt = """You are the Lead Researcher. Execute research for: {research_brief}
Temporal Intent: {temporal_intent}
Complexity Tier: {complexity_tier}
Constraints: {constraints}
Date: {date}

Methodology:
- Generate research plan with specific, verifiable tasks
- Prioritize high-quality, authoritative sources
- Flag potential contradictions immediately
- Maintain 95%+ confidence threshold

Dynamic Parameters:
- Max Concurrent Units: {max_concurrent_research_units}
- Tool Budget: {max_researcher_iterations}
- MCP Override: {mcp_prompt}

Output structured research nodes with confidence scores."""

reasoning_council_prompt = """You are the Reasoning Council. Synthesize perspectives on: {topic}
Context: {context}

Evaluate from these perspectives:
{perspectives}

Apply rigorous logical consistency checks:
- Identify contradictory claims
- Assess source credibility chains
- Flag confidence degradation
- Propose resolution strategies

Return coherent synthesis with confidence-weighted conclusions."""

final_report_generation_prompt = """Create executive research report for: {topic}
Findings Summary: {findings}
Confidence Metrics: {confidence}
Contradictions Resolved: {resolved_issues}

Format:
- Executive Summary (200 words max)
- Methodology & Sources
- Key Findings (numbered)
- Confidence Analysis
- Limitations & Risks
- Future Research Directions

Maintain scientific objectivity. Flag any unresolved uncertainties."""

meta_cognitive_router_prompt = """Analyze research brief: {research_brief}
Context: {memory_context}
Date: {date}

Route to optimal research strategy:
- Complexity tier (Simple/Intermediate/Advanced/Expert)
- Dynamic tool budget allocation
- Concurrent unit scaling
- Temporal focus adjustment
- Constraint compliance verification

Return JSON with routing parameters."""

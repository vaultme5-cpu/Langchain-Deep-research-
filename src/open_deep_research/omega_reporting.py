import re
"""I16.13: omega_reporting.py - report rendering + dashboards.
Extracted from deep_researcher.py. Uses lazy imports for the
quota accessors to avoid circular imports at module load."""

NL = chr(10)
def _sanitize_report_citations(content, evidence_count):
    text = str(content or "")
    def replace_invalid(match):
        n = int(match.group(1))
        return match.group(0) if 1 <= n <= evidence_count else ""
    text = re.sub(r"\[(\d+)\]", replace_invalid, text)
    if "Sources" not in text:
        text += NL + NL + "Sources" + NL
    return text.strip()


def _render_final_report(artifact, verified, confidence, consensus):
    valid_ids = set(range(1, len(verified) + 1))
    def refs(ids):
        result = []
        for item in ids or []:
            try: number = int(item)
            except Exception: continue
            if number in valid_ids and number not in result: result.append(number)
        return "".join(" [" + str(x) + "]" for x in result)
    out = []
    title = str(getattr(artifact, "title", "") or "Omega Research Report").strip()
    out.append("# " + title)
    summary = str(getattr(artifact, "executive_summary", "") or "").strip()
    if summary:
        out.extend(["", "## Executive Summary", summary + refs(getattr(artifact, "executive_evidence_ids", []))])
    for section in getattr(artifact, "sections", []) or []:
        heading = str(getattr(section, "heading", "") or "Analysis").strip()
        scontent = str(getattr(section, "content", "") or "").strip()
        if not scontent: continue
        out.extend(["", "## " + heading, scontent + refs(getattr(section, "evidence_ids", []))])
    uncertainties = getattr(artifact, "key_uncertainties", []) or []
    out.extend(["", "## Key Uncertainties"])
    if uncertainties:
        for item in uncertainties:
            if str(item).strip():
                out.append("- " + str(item).strip())
    else:
        out.append("- No material uncertainties identified.")

    out.extend(["", "## Sources"])
    for index, node in enumerate(verified, start=1):
        out.append("[" + str(index) + "] " + str(getattr(node, "title", "") or "Source") + " — " + str(getattr(node, "url", "")))
    out.extend(["", "## Epistemic Audit", str(consensus or "N/A"), "Confidence: " + str(round(float(confidence), 3)), "", "## Watchlist"])
    watchlist = getattr(artifact, "watchlist", []) or []
    if watchlist:
        for item in watchlist:
            if str(item).strip(): out.append("- " + str(item).strip())
    else:
        out.append("- No watchlist items.")
    return NL.join(out).strip()


def _render_epistemic_dashboard(state):
    """H14.1: Render true epistemic state dashboard."""
    evidence_graph = state.get("evidence_graph", [])
    status_counts = {}
    for node in evidence_graph:
        status = str(getattr(node, "epistemic_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    total_nodes = len(evidence_graph)
    confidence = float(state.get("confidence_score", 0.0) or 0.0)
    out = []
    out.append("[EPISTEMIC DASHBOARD]")
    out.append("Evidence nodes: " + str(total_nodes))
    for status, count in sorted(status_counts.items()):
        out.append("  " + status + ": " + str(count))
    out.append("Confidence: " + str(round(confidence, 2)))
    out.append("Research iterations: " + str(state.get("research_iterations", 0)))
    out.append("Research status: " + str(state.get("research_status", "unknown")))
    return NL.join(out)


def _render_tool_health_dashboard():
    """H14.2: Render tool health dashboard from execution health."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _q_execution_health = _dr._q_execution_health
    _q_run_budget = _dr._q_run_budget
    _q_brain_budgets = _dr._q_brain_budgets
    _q_tpm_window = _dr._q_tpm_window
    _q_reservation_ledger = _dr._q_reservation_ledger
    health = _q_execution_health()
    out = []
    out.append("[TOOL HEALTH DASHBOARD]")
    out.append("Status: " + str(health.get("status", "UNKNOWN")))
    out.append("Warnings: " + str(len(health.get("warnings", []))))
    out.append("Failures: " + str(len(health.get("failures", []))))
    out.append("Fallbacks: " + str(len(health.get("fallbacks", []))))
    all_events = health.get("warnings", []) + health.get("failures", []) + health.get("fallbacks", [])
    for event in all_events[-5:]:
        out.append("  " + str(event)[:70])
    return NL.join(out)


def _render_budget_dashboard():
    """H14.3: Render budget dashboard."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _q_execution_health = _dr._q_execution_health
    _q_run_budget = _dr._q_run_budget
    _q_brain_budgets = _dr._q_brain_budgets
    _q_tpm_window = _dr._q_tpm_window
    _q_reservation_ledger = _dr._q_reservation_ledger
    out = []
    out.append("[BUDGET DASHBOARD]")
    run_used = float(_q_run_budget().get("used", 0.0))
    run_cap = float(_q_run_budget().get("cap", 0.0))
    pct = (run_used / max(1, run_cap)) * 100
    out.append("Run budget: " + str(round(run_used)) + " / " + str(round(run_cap)) + " tokens (" + str(round(pct, 1)) + "%)")
    for brain, budget in _q_brain_budgets().items():
        b_used = float(budget.get("used", 0.0))
        b_cap = float(budget.get("cap", 0.0))
        b_pct = (b_used / max(1, b_cap)) * 100
        out.append("  " + str(brain) + ": " + str(round(b_used)) + " / " + str(round(b_cap)) + " (" + str(round(b_pct, 1)) + "%)")
    import time as _t
    now = _t.time()
    active_tpm = sum(e[1] for e in _q_tpm_window() if now - e[0] < 60.0)
    out.append("TPM (60s window): " + str(round(active_tpm)))
    active_res = sum(1 for r in _q_reservation_ledger() if r.get("status") == "active")
    out.append("Active reservations: " + str(active_res))
    out.append("Accounting degraded: " + str(_q_run_budget().get("accounting_degraded", False)))
    return NL.join(out)


def _render_research_frontier(state):
    """H14.4: Render research frontier visualization."""
    plan = state.get("research_plan", [])
    completed = set(str(x) for x in state.get("completed_nodes", []))
    out = []
    out.append("[RESEARCH FRONTIER]")
    for node in plan:
        if isinstance(node, dict):
            nid = str(node.get("node_id", "?"))
            deps = node.get("depends_on", []) or []
            if nid in completed:
                status = "[DONE]"
            elif all(str(d) in completed for d in deps):
                status = "[READY]"
            else:
                status = "[BLOCKED]"
            out.append("  " + status + " " + nid + ": " + str(node.get("topic", ""))[:50])
    return NL.join(out)


def _render_full_dashboard(state):
    """H14.1-4: Render complete observability dashboard."""
    sections = [
        _render_epistemic_dashboard(state),
        _render_tool_health_dashboard(),
        _render_budget_dashboard(),
        _render_research_frontier(state),
    ]
    return (NL + NL).join(sections)


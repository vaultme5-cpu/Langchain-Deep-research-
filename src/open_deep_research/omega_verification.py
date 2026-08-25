"""I15.12: omega_verification.py - epistemic gate core.
Extracted from deep_researcher.py. Pure functions, no globals,
no dependency on deep_researcher (no circular import).
"""

def _i8_epistemic_quality_score(evidence_nodes):
    """I13.4: Quality score from canonical verification status."""
    if not evidence_nodes:
        return 0.0

    weights = {
        "clear_support": 1.0,
        "partial_support": 0.6,
        "ambiguous": 0.3,
        "unsupported": 0.0,
        "contradictory": 0.0,
        "unverified": 0.1,
        "quarantined": 0.0,
    }

    total_weight = 0.0

    for node in evidence_nodes:
        status = str(
            getattr(node, "verification_status", "") or ""
        ).strip().lower()

        total_weight += weights.get(status, 0.0)

    return round(total_weight / len(evidence_nodes), 3)

def _i8_adjust_confidence(base_confidence, quality_score, contradiction_count=0):
    """I13.4: Adjust confidence based on evidence quality and contradictions."""
    adjusted = float(base_confidence)
    reasons = []
    if quality_score < 0.5:
        penalty = (0.5 - quality_score) * 0.4
        adjusted -= penalty
        reasons.append("quality_penalty:" + str(round(penalty, 3)))
    elif quality_score > 0.8:
        bonus = (quality_score - 0.8) * 0.1
        adjusted += bonus
        reasons.append("quality_bonus:" + str(round(bonus, 3)))
    if contradiction_count > 0:
        penalty = min(contradiction_count * 0.05, 0.2)
        adjusted -= penalty
        reasons.append("contradiction_penalty:" + str(round(penalty, 3)))
    adjusted = max(0.0, min(1.0, adjusted))
    return round(adjusted, 3), "; ".join(reasons) if reasons else "no_adjustment"

def _i8_report_eligibility(evidence_nodes, adjusted_confidence, min_confidence=0.4):
    """I13.4: Determine if evidence quality supports report generation."""
    if not evidence_nodes:
        return False, "no_evidence"
    quality = _i8_epistemic_quality_score(evidence_nodes)
    unverified_ratio = sum(
        1
        for n in evidence_nodes
        if str(
            getattr(n, "verification_status", "") or ""
        ).strip().upper() in (
            "UNVERIFIED",
            "AMBIGUOUS",
            "UNSUPPORTED",
        )
    ) / max(1, len(evidence_nodes))
    contradicted_count = sum(1 for n in evidence_nodes if getattr(n, "contradicts", []))
    poisoned_count = sum(1 for n in evidence_nodes if "[QUARANTINED" in str(getattr(n, "claim", "") or ""))
    if adjusted_confidence < min_confidence:
        return False, "confidence_below_threshold:" + str(round(adjusted_confidence, 3))
    if unverified_ratio > 0.7:
        return False, "excessive_unverified:" + str(round(unverified_ratio, 2))
    if contradicted_count > len(evidence_nodes) * 0.5:
        return False, "majority_contradicted"
    if poisoned_count > 0:
        return False, "poisoned_evidence_present"
    if quality < 0.3:
        return False, "quality_too_low:" + str(quality)
    return True, "eligible:quality=" + str(quality)

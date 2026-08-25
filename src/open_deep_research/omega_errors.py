"""I14.16 step 1: omega_errors.py - error semantics module.
Extracted from deep_researcher.py (pure functions only).
No globals. No dependency on deep_researcher (no circular import).
_record_health_event stays in deep_researcher.py (needs quota context).
Behavior preserved exactly - no functional changes.
"""

_I9_ERROR_TAXONOMY = {
    "FATAL": {"halt": True, "deliver_output": False, "severity": 3},
    "ACCOUNTING_CORRUPTION": {"halt": True, "deliver_output": False, "retry": False, "severity": 3},
    "SECURITY_VIOLATION": {"halt": True, "deliver_output": False, "retry": False, "severity": 3},
    "UNSAFE": {"halt": False, "deliver_output": False, "severity": 2},
    "DEGRADED": {"halt": False, "deliver_output": True, "severity": 1},
    "BENIGN": {"halt": False, "deliver_output": True, "severity": 0},
}

_I9_VALID_CONTEXTS = frozenset({
    "general", "llm", "tool", "research", "security", "quota",
    "memory", "verification", "report", "dag", "session", "intake",
})


def _i9_classify_error(error, context="general"):
    error_str = str(error).lower()
    fatal_keywords = ["auth", "key pool exhausted", "budget exhausted", "all brains",
                      "run budget", "brain budget", "ledger corruption", "api key"]
    if any(k in error_str for k in fatal_keywords):
        return "FATAL"
    unsafe_keywords = ["injection", "poison", "quarantine", "untrusted", "malicious",
                       "exfiltration", "prompt injection"]
    if any(k in error_str for k in unsafe_keywords):
        return "UNSAFE"
    if context == "security":
        return "UNSAFE"
    if context in ("tool", "research"):
        return "DEGRADED"
    degraded_keywords = ["rate_limit", "timeout", "server_error", "fallback",
                         "retry", "truncated", "partial"]
    if any(k in error_str for k in degraded_keywords):
        return "DEGRADED"
    return "DEGRADED"


def _i9_error_action(error_class):
    return _I9_ERROR_TAXONOMY.get(error_class, _I9_ERROR_TAXONOMY["DEGRADED"])


def _i9_should_deliver_output(error_class):
    action = _i9_error_action(error_class)
    return action["deliver_output"]


def _i9_should_halt(error_class):
    action = _i9_error_action(error_class)
    return action["halt"]


def _i9_should_deliver(error_class):
    return _i9_error_action(error_class)["deliver_output"]


def _i9_should_retry(error_class):
    """Return the explicit retry policy for an error class.

    Error classes that do not explicitly opt into retrying are treated
    as non-retryable. This prevents missing taxonomy fields from causing
    a KeyError during error handling.
    """
    action = _i9_error_action(error_class)
    return bool(action.get("retry", False))


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
    if "brain budget" in s or "budget exhausted" in s: return "BUDGET_EXHAUSTED"
    if "tpm_exhausted" in s or "tpm exhausted" in s: return "TPM_EXHAUSTED"
    if "run_budget_exhausted" in s or "run token budget" in s: return "RUN_BUDGET_EXHAUSTED"
    if "brain_budget_exhausted" in s: return "BRAIN_BUDGET_EXHAUSTED"
    if "ledger_corruption" in s or "ledger corruption" in s: return "LEDGER_CORRUPTION"
    return "UNKNOWN"


class _I13_12_HaltExecution(Exception):
    def __init__(self, error_class, reason):
        self.error_class = error_class
        self.reason = reason
        super().__init__("HALT [" + str(error_class) + "]: " + str(reason))


def _i13_12_enforce_policy(error, context="general"):
    cls = _i9_classify_error(error, context)
    action = _i9_error_action(cls)
    if action["halt"]:
        raise _I13_12_HaltExecution(cls, str(error)[:120])
    return cls, action


def _run_omega_errors_self_test():
    passed = 0
    failed = 0
    details = []
    def check(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            details.append("FAIL: " + name)
    check("classify injection -> UNSAFE", _i9_classify_error("prompt injection detected", "general") == "UNSAFE")
    check("classify api key -> FATAL", _i9_classify_error("api key invalid", "general") == "FATAL")
    check("classify security ctx -> UNSAFE", _i9_classify_error("weird thing", "security") == "UNSAFE")
    check("classify tool ctx -> DEGRADED", _i9_classify_error("weird thing", "tool") == "DEGRADED")
    check("action FATAL halts", _i9_error_action("FATAL")["halt"] == True)
    check("action BENIGN delivers", _i9_error_action("BENIGN")["deliver_output"] == True)
    check("should_halt FATAL", _i9_should_halt("FATAL") == True)
    check("should_halt BENIGN false", _i9_should_halt("BENIGN") == False)
    check("should_deliver UNSAFE false", _i9_should_deliver("UNSAFE") == False)
    check("should_retry ACCOUNTING_CORRUPTION", _i9_should_retry("ACCOUNTING_CORRUPTION") == False)
    check("model error 429 -> RATE_LIMIT", classify_model_error("429 rate limit exceeded") == "RATE_LIMIT")
    check("model error 401 -> AUTH", classify_model_error("401 unauthorized") == "AUTH")
    check("model error mystery -> UNKNOWN", classify_model_error("mystery") == "UNKNOWN")
    cls, act = _i13_12_enforce_policy("minor hiccup", "general")
    check("enforce benign returns DEGRADED", cls == "DEGRADED")
    raised = False
    try:
        _i13_12_enforce_policy("api key invalid", "general")
    except _I13_12_HaltExecution:
        raised = True
    check("enforce fatal raises Halt", raised == True)
    return {"passed": passed, "failed": failed, "details": details,
            "success": failed == 0, "verdict": "PASS" if failed == 0 else "FAIL"}


if __name__ == "__main__":
    r = _run_omega_errors_self_test()
    print("omega_errors self-test: " + str(r["passed"]) + " passed, " + str(r["failed"]) + " failed")
    for d in r["details"]:
        print(d)
    print("VERDICT: " + r["verdict"])

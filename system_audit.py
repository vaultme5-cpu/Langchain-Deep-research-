"""Omega Supremacy System Audit (zero-token CI gate)."""
import importlib
import os
import py_compile
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(ROOT, "src", "open_deep_research")
FILES = ["configuration.py", "state.py", "prompts.py", "utils.py", "deep_researcher.py", "omega_ui.py"]

def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print("[" + status + "] " + label + (": " + str(detail) if detail else ""))
    return ok

def main():
    results = []
    results.append(check("Python", True, sys.version.split(" ")[0]))
    results.append(check("Package dir", os.path.isdir(PKG_DIR), PKG_DIR))
    src_dir = os.path.join(ROOT, "src")
    for p in (src_dir, PKG_DIR):
        if os.path.isdir(p) and p not in sys.path: sys.path.insert(0, p)
    results.append(check("requirements.txt", os.path.isfile(os.path.join(ROOT, "requirements.txt"))))
    groq = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or ""
    gem = os.environ.get("GEMINI_API_KEY") or ""
    print("[WARN] Groq key: " + ("Online" if groq.strip() else "Offline"))
    print("[WARN] Gemini key: " + ("Online" if gem.strip() else "Offline"))
    for f in FILES:
        p = os.path.join(PKG_DIR, f) if f != "omega_ui.py" else os.path.join(ROOT, f)
        if not os.path.isfile(p):
            results.append(check("exists: " + f, False, p))
            continue
        try:
            py_compile.compile(p, doraise=True)
            results.append(check("syntax: " + f, True))
        except Exception as e:
            results.append(check("syntax: " + f, False, str(e)))
    for m in ["open_deep_research.configuration", "open_deep_research.state", "open_deep_research.prompts", "open_deep_research.utils", "open_deep_research.deep_researcher"]:
        try:
            importlib.import_module(m)
            results.append(check("import: " + m, True))
        except Exception as e:
            results.append(check("import: " + m, False, str(e)))
    try:
        mod = importlib.import_module("open_deep_research.deep_researcher")
        results.append(check("graph object", getattr(mod, "deep_researcher", None) is not None))
        results.append(check("breaker present", hasattr(mod, "_BRAIN_HEALTH")))
        results.append(check("budget present", hasattr(mod, "_RUN_BUDGET")))
    except Exception as e:
        results.append(check("graph object", False, str(e)))
    failed = len([r for r in results if not r])
    print("")
    if failed == 0:
        print("AUDIT RESULT: ALL CRITICAL CHECKS PASSED")
    else:
        print("AUDIT RESULT: " + str(failed) + " CRITICAL CHECKS FAILED")
        raise SystemExit(1)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("AUDIT CRASHED")
        print(traceback.format_exc())
        raise SystemExit(1)

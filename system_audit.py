"""Omega Supremacy System Audit."""
import importlib
import os
import py_compile
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(ROOT, "src", "open_deep_research")
FILES = ["configuration.py", "state.py", "prompts.py", "utils.py", "deep_researcher.py"]

def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print("[" + status + "] " + label + (": " + str(detail) if detail else ""))
    return ok

def warn(label, detail=""):
    print("[WARN] " + label + (": " + str(detail) if detail else ""))

def main():
    results = []
    py = sys.version.split(" ")[0]
    results.append(check("Python version", True, py))
    results.append(check("Package directory exists", os.path.isdir(PKG_DIR), PKG_DIR))
    init_path = os.path.join(PKG_DIR, "__init__.py")
    results.append(check("__init__.py exists", os.path.isfile(init_path), init_path))
    src_dir = os.path.join(ROOT, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if os.path.isdir(PKG_DIR) and PKG_DIR not in sys.path:
        sys.path.insert(0, PKG_DIR)
    req_path = os.path.join(ROOT, "requirements.txt")
    results.append(check("requirements.txt exists", os.path.isfile(req_path), req_path))
    groq = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or ""
    jina = os.environ.get("JINA_API_KEY") or ""
    if groq.strip(): warn("Groq key present", "Online")
    else: warn("Groq key present", "Offline")
    if jina.strip(): warn("Jina key present", "Online")
    else: warn("Jina key present", "Offline")
    for f in FILES:
        p = os.path.join(PKG_DIR, f)
        if not os.path.isfile(p):
            results.append(check("File exists: " + f, False, p))
            continue
        try:
            py_compile.compile(p, doraise=True)
            results.append(check("Syntax: " + f, True))
        except Exception as e:
            results.append(check("Syntax: " + f, False, str(e)))
    ui_path = os.path.join(ROOT, "omega_ui.py")
    if os.path.isfile(ui_path):
        try:
            py_compile.compile(ui_path, doraise=True)
            results.append(check("Syntax: omega_ui.py", True))
        except Exception as e:
            results.append(check("Syntax: omega_ui.py", False, str(e)))
    else:
        results.append(check("omega_ui.py exists", False, ui_path))
    mods = [
        "open_deep_research.configuration",
        "open_deep_research.state",
        "open_deep_research.prompts",
        "open_deep_research.utils",
        "open_deep_research.deep_researcher",
    ]
    for m in mods:
        try:
            importlib.import_module(m)
            results.append(check("Import: " + m, True))
        except Exception as e:
            results.append(check("Import: " + m, False, str(e)))
    try:
        mod = importlib.import_module("open_deep_research.deep_researcher")
        graph = getattr(mod, "deep_researcher", None)
        results.append(check("Graph object exists", graph is not None))
    except Exception as e:
        results.append(check("Graph object exists", False, str(e)))
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

import hashlib, re, io, asyncio, contextlib, os, json, math, sqlite3, time, threading, logging
from datetime import datetime
from typing import List, Optional, Any
from urllib.parse import quote, quote_plus
import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import filter_messages
from langchain_core.tools import tool
from open_deep_research.configuration import Configuration, SearchAPI

try:
    import arxiv
except ImportError:
    arxiv = None

TOKEN_RE = "[a-z0-9_]{4,}"
NL = chr(10)
MAX_TOOL_TEXT = 6000

def _current_loop():
    try: return asyncio.get_running_loop()
    except RuntimeError:
        try: return asyncio.get_event_loop()
        except RuntimeError: return asyncio.new_event_loop()

class LoopSafeSemaphore:
    def __init__(self, limit):
        self.limit = limit
        self._semaphores = {}
    async def __aenter__(self):
        loop = _current_loop()
        key = id(loop)
        if key not in self._semaphores: self._semaphores[key] = asyncio.Semaphore(self.limit)
        await self._semaphores[key].__aenter__()
        return self
    async def __aexit__(self, exc_type, exc, tb):
        loop = _current_loop()
        sem = self._semaphores.get(id(loop))
        if sem is not None: await sem.__aexit__(exc_type, exc, tb)

class GroqShield:
    def __init__(self, keys):
        self.keys = keys if keys else [""]
        self.cooldowns = {k: 0.0 for k in self.keys}
        self.lock = threading.Lock()
        self.idx = 0
        self.last = None
    def get_key(self, last_failed=None):
        with self.lock:
            now = time.time()
            if last_failed and last_failed in self.cooldowns: self.cooldowns[last_failed] = now + 60.0
            avail = [k for k in self.keys if self.cooldowns.get(k, 0.0) <= now]
            if not avail: return min(self.keys, key=lambda k: self.cooldowns.get(k, 0.0))
            self.idx = (self.idx + 1) % len(avail)
            return avail[self.idx]

_raw_groq = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
_groq_keys = [k.strip() for k in _raw_groq.split(",") if k.strip()]
_shield = GroqShield(_groq_keys)

def get_api_key_for_model(model_name, config, last_failed=None):
    m = str(model_name or "").lower()
    if m.startswith("groq:"): return _shield.get_key(last_failed)

    return None

def get_today_str(): return datetime.now().strftime("%a %b %d, %Y")

def get_config_value(value):
    if value is None: return None
    return value.value if hasattr(value, "value") else value

def _domain_from_url(url):
    try: return str(url).split("//")[-1].split("/")[0].replace("www.", "")
    except Exception: return ""

def _safe_url(url):
    try:
        from urllib.parse import urlparse
        import ipaddress
        p = urlparse(str(url))
        if p.scheme not in ("http", "https"): return False
        h = (p.hostname or "").lower()
        if h == "localhost" or h.endswith(".local") or "metadata.google.internal" in h: return False
        try:
            ip = ipaddress.ip_address(h)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved: return False
        except Exception:
            pass
        return True
    except Exception:
        return False


# ============================================================
# I14.9: REDIRECT-SAFE URL VALIDATION
# Every redirect hop is validated. Final destination must be
# safe BEFORE content is accepted. SSRF protection.
# ============================================================
_I14_9_MAX_REDIRECTS = 5
_I14_9_BLOCKED_HOSTS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "metadata.google.internal", "metadata.aws.amazon.com",
    "169.254.169.254",  # cloud metadata
})

def _i14_9_is_safe_ip(ip_str):
    """I14.9: Validate an IP address is not private/loopback/link-local/reserved/metadata."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(str(ip_str))
    except Exception:
        return False, "unparseable_ip"
    if ip.is_loopback:
        return False, "loopback"
    if ip.is_private:
        return False, "private"
    if ip.is_link_local:
        return False, "link_local"
    if ip.is_reserved:
        return False, "reserved"
    if ip.is_multicast:
        return False, "multicast"
    # Explicit metadata endpoint check
    if str(ip) == "169.254.169.254":
        return False, "cloud_metadata"
    return True, "safe"

def _i14_9_validate_host(hostname):
    """I14.9: Resolve hostname and validate all resolved IPs are safe.
    Returns (is_safe, reason). Prevents DNS rebinding."""
    import socket
    host = str(hostname or "").lower().strip()
    if not host:
        return False, "empty_host"
    if host in _I14_9_BLOCKED_HOSTS:
        return False, "blocked_host"
    if host.endswith(".local") or host.endswith(".internal"):
        return False, "internal_domain"
    if host.endswith(".onion") or host.endswith(".i2p"):
        return False, "anonymity_network"
    # Try to parse as IP directly
    import ipaddress
    try:
        ipaddress.ip_address(host)
        return _i14_9_is_safe_ip(host)
    except ValueError:
        pass  # Not an IP, resolve it
    # DNS resolution
    try:
        addrinfo = socket.getaddrinfo(host, None)
    except Exception:
        return False, "dns_resolution_failed"
    for family, socktype, proto, canonname, sockaddr in addrinfo:
        ip_str = sockaddr[0]
        is_safe, reason = _i14_9_is_safe_ip(ip_str)
        if not is_safe:
            return False, "resolved_unsafe_ip:" + reason
    return True, "safe"

def _i14_9_validate_url_deep(url):
    """I14.9: Full URL validation including DNS resolution.
    Returns (is_safe, reason)."""
    from urllib.parse import urlparse
    url_str = str(url or "").strip()
    if not url_str:
        return False, "empty_url"
    try:
        parsed = urlparse(url_str)
    except Exception:
        return False, "parse_error"
    if parsed.scheme not in ("http", "https"):
        return False, "non_http_scheme"
    host = parsed.hostname or ""
    return _i14_9_validate_host(host)

async def _i14_9_safe_follow(client, url, max_redirects=None, **kwargs):
    """I14.9: Follow redirects with per-hop validation.
    Returns the final response. Raises ValueError on unsafe redirect."""
    if max_redirects is None:
        max_redirects = _I14_9_MAX_REDIRECTS
    current_url = str(url)
    for hop in range(max_redirects + 1):
        # Validate current URL before requesting
        is_safe, reason = _i14_9_validate_url_deep(current_url)
        if not is_safe:
            raise ValueError("I14.9 unsafe URL at hop " + str(hop) + ": " + reason)
        # Make request WITHOUT following redirects
        response = await client.get(current_url, follow_redirects=False, **kwargs)
        # Check if redirect
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location", "")
            if not location:
                raise ValueError("I14.9 redirect without location header")
            # Resolve relative URLs
            from urllib.parse import urljoin
            next_url = urljoin(current_url, location)
            # Validate the redirect target BEFORE following
            redirect_safe, redirect_reason = _i14_9_validate_url_deep(next_url)
            if not redirect_safe:
                raise ValueError("I14.9 unsafe redirect target: " + redirect_reason)
            current_url = next_url
            continue
        # Not a redirect — return the response
        return response
    raise ValueError("I14.9 max redirects exceeded: " + str(max_redirects))


# ============================================================
# I14.10: TYPED TOOL RESULT SYSTEM
# Failed tool calls NEVER become evidence.
# ============================================================
_I14_10_TOOL_STATUSES = frozenset({"SUCCESS", "DEGRADED", "FAILED", "QUARANTINED"})

def _i14_10_classify_tool_output(output, tool_name="tool"):
    """I14.10: Classify tool output into typed status.
    Returns (status, content). Status determines evidence eligibility."""
    text = str(output or "")
    if not text.strip():
        return "FAILED", ""
    # Quarantine markers (from G13.1/G14.3)
    if "[QUARANTINED" in text:
        return "QUARANTINED", text
    # Failure/fallback markers
    failure_markers = ["[FALLBACK]", "[JINA FALLBACK]", "[PDF] Download failed",
                       "[PDF] Unable to extract", "AUDIT_FAILED", "[PDF] Unsafe URL"]
    for marker in failure_markers:
        if marker in text:
            return "FAILED", text
    # Degraded markers (partial results)
    degraded_markers = ["[TRUNCATED]", "[CHUNK", "No results", "No GitHub repos",
                        "No HF models", "No papers found", "No Wikipedia results",
                        "No Searxng results", "Jina failed", "Searxng failed",
                        "Wikipedia failed", "GitHub failed", "HF failed", "ArXiv failed"]
    for marker in degraded_markers:
        if marker in text:
            return "DEGRADED", text
    return "SUCCESS", text

def _i14_10_wrap_tool_result(
    status,
    source,
    content,
    error_class=None,
    source_result_id=None,
    final_url=None,
):
    """I14.10: Build a typed tool result dict.

    Compatibility wrapper for older callers.
    source_result_id and final_url are optional so existing
    call sites remain backward compatible.
    """
    import time as _t
    import hashlib as _h

    now = _t.time()
    request_id = _h.sha256(
        (str(source) + str(now)).encode("utf-8")
    ).hexdigest()[:12]

    return {
        "status": str(status or "FAILED"),
        "source": str(source or "unknown"),
        "content": str(content or ""),
        "error_class": error_class,
        "request_id": request_id,
        "retrieved_at": now,
        "source_result_id": source_result_id,
        "final_url": final_url,
    }

def _i14_10_can_enter_evidence(status):
    """I14.10: Determine if a tool result status allows evidence entry."""
    if status == "SUCCESS":
        return True
    if status == "DEGRADED":
        return True  # enters with degraded provenance
    # FAILED and QUARANTINED never enter evidence
    return False

def _tokens(text): return set(re.findall(TOKEN_RE, str(text or "").lower()))

@tool(description="Search GitHub repositories.")
def github_sniper(query: str) -> str:
    try:
        h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Omega/2.0"}
        r = httpx.get("https://api.github.com/search/repositories", headers=h, params={"q": query, "sort": "stars", "per_page": 5}, timeout=15)
        r.raise_for_status()
        items = r.json().get("items", [])
        return _i15_7_make_tool_result("DEGRADED", "github_sniper", "No GitHub repos found.")
        return _i15_7_make_tool_result("SUCCESS", "github_sniper", (NL + NL).join(["[ " + str(i+1) + "] " + str(x.get("full_name")) + " | Stars: " + str(x.get("stargazers_count")) + NL + "URL: " + str(x.get("html_url")) for i, x in enumerate(items)]))
    except Exception as e: return _i15_7_make_tool_result("FAILED", "github_sniper", f"GitHub failed: {e}", error_class="GITHUB_ERROR")

@tool(description="Search HuggingFace models.")
def huggingface_sniper(query: str) -> str:
    try:
        r = httpx.get("https://huggingface.co/api/models", params={"search": query, "sort": "downloads", "direction": "-1", "limit": 5}, timeout=15)
        r.raise_for_status()
        models = r.json()
        if not models: return "No HF models found."
        return (NL + NL).join(["[" + str(i+1) + "] " + str(m.get("modelId")) + " | DL: " + str(m.get("downloads", 0)) + NL + "URL: https://huggingface.co/" + str(m.get("modelId")) for i, m in enumerate(models)])
    except Exception as e: return f"HF failed: {e}"

@tool(description="Search ArXiv papers.")
def arxiv_search(query: str) -> str:
    if not arxiv: return "ArXiv not installed."
    try:
        s = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        out = []
        for i, p in enumerate(s.results()):
            out.append("[" + str(i+1) + "] " + str(p.title) + " (" + str(p.entry_id) + ")" + NL + str(p.summary)[:800])
        return (NL + NL).join(out) if out else "No papers found."
    except Exception: return "ArXiv failed."


# ============================================================
# I14.12: RESTRICTED PYTHON EVALUATOR HARDENING
# Deterministic computation only. NOT an OS-level sandbox.
# ============================================================
_I14_12_MAX_OUTPUT_CHARS = 50000
_I14_12_MAX_LOOP_RANGE = 10000000
_I14_12_MAX_ALLOCATION = 10000000

def _i14_12_check_code_safety(tree):
    """I14.12: Detect attack patterns in AST. Returns (is_safe, reason)."""
    import ast
    func_names = set()
    for _sn in ast.walk(tree):
        if isinstance(_sn, ast.FunctionDef):
            func_names.add(_sn.name)
    for _sn in ast.walk(tree):
        if isinstance(_sn, ast.Attribute):
            attr_chain = []
            current = _sn
            while isinstance(current, ast.Attribute):
                attr_chain.append(current.attr)
                current = current.value
            chain_str = ".".join(reversed(attr_chain))
            if "__class__" in chain_str and ("__bases__" in chain_str or "__subclasses__" in chain_str or "__mro__" in chain_str):
                return False, "object_traversal: " + chain_str[:60]
            if "__globals__" in chain_str or "__code__" in chain_str or "__closure__" in chain_str:
                return False, "forbidden_chain: " + chain_str[:60]
        if isinstance(_sn, ast.BinOp) and isinstance(_sn.op, ast.Mult):
            for operand in (_sn.left, _sn.right):
                if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
                    if operand.value > _I14_12_MAX_ALLOCATION:
                        return False, "giant_allocation: " + str(operand.value)
        if isinstance(_sn, ast.Call):
            if isinstance(_sn.func, ast.Name) and _sn.func.id == "range":
                for arg in _sn.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                        if arg.value > _I14_12_MAX_LOOP_RANGE:
                            return False, "huge_loop: range(" + str(arg.value) + ")"
        if isinstance(_sn, ast.Call):
            if isinstance(_sn.func, ast.Name) and _sn.func.id == "print":
                for arg in _sn.args:
                    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult):
                        for op in (arg.left, arg.right):
                            if isinstance(op, ast.Constant) and isinstance(op.value, (int, float)):
                                if op.value > 100000:
                                    return False, "output_flooding"
        if isinstance(_sn, ast.Call):
            if isinstance(_sn.func, ast.Name) and _sn.func.id in ("getattr", "setattr", "delattr", "hasattr"):
                return False, "reflection_via_" + _sn.func.id
    return True, "safe"

@tool(
    description=(
        "Execute deterministic Python computation only. "
        "Network, filesystem, process execution, imports "
        "and reflection are forbidden."
    )
)
def python_repl(code: str) -> str:
    import ast
    import concurrent.futures

    code = str(code or "")

    if len(code) > 12000:
        return (
            "[RESTRICTED EVALUATOR] Code size limit exceeded."
        )

    try:
        tree = ast.parse(code)
        # I14.12: explicit attack-pattern detection
        _i14_12_safe, _i14_12_reason = _i14_12_check_code_safety(tree)
        if not _i14_12_safe:
            return "[RESTRICTED EVALUATOR BLOCKED] " + _i14_12_reason

        if len(list(ast.walk(tree))) > 500:
            return (
                "[RESTRICTED EVALUATOR] AST complexity limit exceeded."
            )

        forbidden_names = {
            "__import__",
            "__builtins__",
            "exec",
            "eval",
            "open",
            "compile",
            "globals",
            "locals",
            "vars",
            "getattr",
            "setattr",
            "delattr",
            "breakpoint",
            "input",
            "help",
            "dir",
        }

        forbidden_attributes = {
            "__class__",
            "__bases__",
            "__subclasses__",
            "__globals__",
            "__builtins__",
            "__import__",
            "__code__",
            "__closure__",
        }

        for node in ast.walk(tree):

            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                return (
                    "[RESTRICTED EVALUATOR] Imports are forbidden."
                )

            if (
                isinstance(node, ast.Name)
                and node.id in forbidden_names
            ):
                return (
                    "[RESTRICTED EVALUATOR] Forbidden symbol."
                )

            if (
                isinstance(node, ast.Attribute)
                and node.attr
                in forbidden_attributes
            ):
                return (
                    "[RESTRICTED EVALUATOR] Forbidden attribute."
                )

        def _run():
            out = io.StringIO()

            safe_builtins = {
                "print": print,
                "len": len,
                "range": range,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "isinstance": isinstance,
                "type": type,
            }

            namespace = {
                "__builtins__": safe_builtins,
                "json": json,
                "re": re,
                "math": math,
                "datetime": datetime,
                "BeautifulSoup": BeautifulSoup,
            }

            with contextlib.redirect_stdout(
                out
            ):
                exec(
                    code,
                    namespace,
                    {},
                )

            return (
                out.getvalue()
                or "OK"
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        ) as executor:

            future = executor.submit(
                _run
            )

            try:
                return future.result(
                    timeout=30.0
                )
            except concurrent.futures.TimeoutError:
                return (
                    "[RESTRICTED EVALUATOR] Timeout: execution exceeded time limit."
                )

    except Exception as exc:
        return (
            "[FALLBACK] "
            + str(exc)
        )



@tool(description="Audit URL pricing.")
def audit_pricing(url: str) -> str:
    return _i15_7_make_tool_result("FAILED", "audit_pricing", "AUDIT_FAILED", error_class="AUDIT_FAILED")
    try:
        r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, follow_redirects=True)
        t = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True).lower()
        red = sum(1 for f in ["contact sales", "book a demo", "enterprise pricing"] if f in t)
        grn = sum(1 for f in ["open source", "free tier", "100% free", "mit license"] if f in t)
        if grn > red and grn > 0: return "VERIFIED_FREE: " + url
        if red > 0 and grn == 0: return "PAID_ENTERPRISE: " + url
        return "UNKNOWN: " + url
    except Exception: return "AUDIT_FAILED"

def jina_scraper(url):
    if not _safe_url(url): return "[JINA FALLBACK]"
    key = os.environ.get("JINA_API_KEY", "")
    h = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if key: h["Authorization"] = "Bearer " + key
    try:
        r = httpx.get("https://r.jina.ai/" + quote(str(url), safe=""), headers=h, timeout=30.0)
        r.raise_for_status()
        data = r.json().get("data", {})
        return str(data.get("content", ""))[:MAX_TOOL_TEXT]
    except Exception: return "[JINA FALLBACK]"


# ============================================================
# I16.7: IMMUTABLE SOURCE ARTIFACT CREATION
# Created at retrieval time. Never inferred later.
# ============================================================
def _i16_7_create_source_artifact(url, content, http_status=200, content_type="text/html",
                                   final_url=None, run_id=""):
    """I16.7: Create an immutable SourceArtifact at retrieval time.
    Returns a dict with all required fields. Never infers identity later."""
    import time as _t
    import hashlib as _h
    import re as _re
    url_str = str(url or "").strip()
    content_str = str(content or "")
    # Canonical URL (strip tracking, normalize)
    canonical = url_str.lower().split("?")[0].split("#")[0]
    canonical = _re.sub(r"^https?://", "", canonical)
    canonical = _re.sub(r"^www\.", "", canonical)
    canonical = canonical.rstrip("/")
    # Hashes
    raw_hash = _h.sha256(content_str.encode("utf-8")).hexdigest()
    normalized = _re.sub(r"\s+", " ", content_str.lower()).strip()
    norm_hash = _h.sha256(normalized.encode("utf-8")).hexdigest()
    # Unique source_result_id (deterministic from URL + run_id)
    srid_input = canonical + "|" + str(run_id or "")
    source_result_id = "src_" + _h.sha256(srid_input.encode("utf-8")).hexdigest()[:16]
    return {
        "source_result_id": source_result_id,
        "run_id": str(run_id or ""),
        "canonical_url": canonical,
        "retrieval_timestamp": _t.time(),
        "http_status": int(http_status),
        "content_type": str(content_type or "text/html"),
        "raw_content_hash": raw_hash,
        "normalized_content_hash": norm_hash,
        "final_url": str(final_url or url_str),
        "source_status": "RETRIEVED" if int(http_status) < 400 else "RETRIEVAL_FAILED",
    }


async def validate_urls(urls):
    out = {}
    uniq = list(set([str(u) for u in (urls or []) if str(u).startswith("http")]))[:30]
    if not uniq: return out
    uniq = [u for u in uniq if _safe_url(u)]
    async with httpx.AsyncClient(timeout=5.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
        for u in uniq:
            try:
                resp = await _i14_9_safe_follow(client, u)  # I14.9: redirect-safe
                out[u] = resp.status_code < 400
            except Exception: out[u] = False
    return out

@tool(description="Search web via Jina AI.")
async def jina_search(query: str) -> str:
    key = os.environ.get("JINA_API_KEY", "")
    h = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if key: h["Authorization"] = "Bearer " + key
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get("https://s.jina.ai/" + quote_plus(str(query)), headers=h)
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data: return _i15_7_make_tool_result("DEGRADED", "jina_search", "No results.")
            out = []
            for i, d in enumerate(data[:5]):
                out.append("--- " + str(i+1) + ": " + str(d.get("title")) + " ---" + NL + "URL: " + str(d.get("url")) + NL + str(d.get("content", ""))[:1200])
            return (NL + NL).join(out)
    except Exception as e: return _i15_7_make_tool_result("FAILED", "jina_search", "Jina failed: " + str(e), error_class="JINA_ERROR")

@tool(description="Search web via Searxng.")
async def searxng_search(query: str) -> str:
    base = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(base + "/search", params={"q": query, "format": "json"})
            r.raise_for_status()
            results = r.json().get("results", [])
            return _i15_7_make_tool_result("DEGRADED", "searxng_search", "No Searxng results.")
            out = []
            for i, item in enumerate(results[:5]):
                out.append("--- " + str(i+1) + ": " + str(item.get("title")) + " ---" + NL + "URL: " + str(item.get("url")) + NL + str(item.get("content", ""))[:1200])
            return (NL + NL).join(out)
    except Exception:
        try: return await jina_search.ainvoke(query)
        except Exception: return _i15_7_make_tool_result("FAILED", "searxng_search", "Searxng failed.", error_class="SEARXNG_ERROR")

@tool(description="Search Wikipedia.")
def wikipedia_rest_search(query: str) -> str:
    try:
        r = httpx.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3}, timeout=10).json()
        hits = r.get("query", {}).get("search", [])
        return _i15_7_make_tool_result("DEGRADED", "wikipedia_rest_search", "No Wikipedia results.")
        out = []
        for h in hits:
            title = str(h.get("title", "")).replace(" ", "_")
            out.append("TITLE: " + str(h.get("title", "")) + NL + "URL: https://en.wikipedia.org/wiki/" + title + NL + str(h.get("snippet", "")))
        return (NL + NL).join(out)
    except Exception: return _i15_7_make_tool_result("FAILED", "wikipedia_rest_search", "Wikipedia failed.", error_class="WIKIPEDIA_ERROR")

@tool(description="Strategic reflection.")
def think_tool(reflection: str) -> str: return "Reflection: " + str(reflection)

@tool("ResearchComplete", description="Signal that the research plan is complete.")
def research_complete_tool() -> str: return "Research complete."

_pdf_semaphore = LoopSafeSemaphore(1)

@tool(description="Ingest PDF via local or Jina fallback.")
async def omega_pdf_ingestor(url: str) -> str:
    return _i15_7_make_tool_result("FAILED", "omega_pdf_ingestor", "Not a PDF.", error_class="NOT_PDF")
    return _i15_7_make_tool_result("FAILED", "omega_pdf_ingestor", "[PDF] Unsafe URL.", error_class="UNSAFE_URL")
    async with _pdf_semaphore:
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                r = await client.get(url)
                r.raise_for_status()
                pdf = r.content
        except Exception as e: return _i15_7_make_tool_result("FAILED", "omega_pdf_ingestor", "[PDF] Download failed: " + str(e), error_class="PDF_DOWNLOAD_ERROR")
        try:
            import pymupdf4llm, tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
                t.write(pdf)
                tp = t.name
            md = pymupdf4llm.to_markdown(tp)
            os.unlink(tp)
            if len(md.strip()) > 200: return "[TIER1 LOCAL]" + NL + md[:15000]
        except Exception: pass
        try:
            txt = jina_scraper(url)
            if txt and not txt.startswith("[JINA"): return "[TIER2 JINA]" + NL + txt
        except Exception: pass
        return _i15_7_make_tool_result("FAILED", "omega_pdf_ingestor", "[PDF] Unable to extract text.", error_class="PDF_EXTRACT_FAILED")

import tempfile as _tmp
OMEGA_MEM = os.path.join(_tmp.gettempdir(), "omega_memory.json")

class OmegaMemory:
    def __init__(self):
        self.data = {"domains": {}}
        try:
            if os.path.exists(OMEGA_MEM):
                with open(OMEGA_MEM, "r", encoding="utf-8") as f: self.data = json.load(f)
        except Exception: pass
    def save(self):
        try:
            t = OMEGA_MEM + ".tmp"
            with open(t, "w", encoding="utf-8") as f: json.dump(self.data, f)
            os.replace(t, OMEGA_MEM)
        except Exception: pass
    def update_domain(self, d, ok):
        if not d: return
        x = self.data.get("domains", {}).get(d, {"trust": 0.5, "hits": 0})
        x["hits"] += 1
        x["trust"] = max(0.0, min(1.0, x.get("trust", 0.5) * 0.9 + (0.1 if ok else -0.05)))
        self.data.setdefault("domains", {})[d] = x
        self.save()
    def get_context_prompt(self):
        top = sorted(self.data.get("domains", {}).items(), key=lambda x: x[1].get("trust", 0.0), reverse=True)[:5]
        if not top: return ""
        return "Trusted: " + ", ".join([str(d) + "(" + str(round(v.get("trust", 0.0), 1)) + ")" for d, v in top])

try: omega_memory = OmegaMemory()
except Exception:
    class _NoOp:
        def get_context_prompt(self): return ""
        def update_domain(self, d, s): pass
    omega_memory = _NoOp()

DB_PATH = os.path.join(_tmp.gettempdir(), "omega.db")

class LocalSQLiteMemory:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.lock = threading.Lock()
        self.c = self.conn.cursor()
        self.c.execute("CREATE TABLE IF NOT EXISTS ev (id INTEGER PRIMARY KEY, claim TEXT, url TEXT)")
        self.conn.commit()
    def store(self, claim, url):
        if not claim or not url: return
        try:
            with self.lock:
                self.c.execute("INSERT INTO ev (claim, url) VALUES (?,?)", (str(claim), str(url)))
                self.conn.commit()
        except Exception: pass
    def recall(self, q, limit=5):
        if not q: return []
        try:
            with self.lock:
                self.c.execute("SELECT claim, url FROM ev WHERE claim LIKE ? ORDER BY id DESC LIMIT ?", ("%" + str(q) + "%", int(limit)))
                return [{"claim": r[0], "url": r[1]} for r in self.c.fetchall()]
        except Exception: return []

try: omega_local_memory = LocalSQLiteMemory()
except Exception:
    class _NoOpDB:
        def store(self, c, u): pass
        def recall(self, q, l=5): return []
    omega_local_memory = _NoOpDB()

_citation_semaphore = LoopSafeSemaphore(5)

# ============================================================
# I13.9: PARALLEL PROGRAMMATIC CITATION VERIFICATION
# Bounded concurrency + per-URL timeout + cache + order kept.
# ============================================================
_VERIFY_CONCURRENCY = 6
_VERIFY_URL_TIMEOUT = 10.0
_VERIFY_URL_CACHE = {}
_VERIFY_CACHE_MAX = 500

def _verify_cache_get(url):
    """I13.9: Cached per-URL verification result, or None."""
    return _VERIFY_URL_CACHE.get(url)

def _verify_cache_put(url, entry):
    """I13.9: Store per-URL result in a bounded cache (FIFO eviction)."""
    if len(_VERIFY_URL_CACHE) >= _VERIFY_CACHE_MAX:
        for k in list(_VERIFY_URL_CACHE.keys())[:_VERIFY_CACHE_MAX // 4]:
            _VERIFY_URL_CACHE.pop(k, None)
    _VERIFY_URL_CACHE[url] = entry


def _sentence_chunks(text):
    clean = re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip()

    if not clean:
        return []

    return [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+",
            clean,
        )
        if part.strip()
    ]


def _verification_numbers(text):
    values = []

    for raw in re.findall(
        r"[0-9]+(?:\.[0-9]+)?",
        str(text or ""),
    ):
        try:
            values.append(float(raw))
        except Exception:
            pass

    return values


def _numbers_supported(
    claim,
    evidence,
):
    wanted = _verification_numbers(
        claim
    )

    if not wanted:
        return True

    found = _verification_numbers(
        evidence
    )

    for value in wanted:
        if not any(
            abs(value - other)
            <= max(
                0.01,
                abs(value) * 0.01,
            )
            for other in found
        ):
            return False

    return True


def _source_kind(url):
    """I14.11: Return canonical SourceKind enum value."""
    domain = _domain_from_url(url).lower()
    if any(x in domain for x in [".gov", ".mil"]):
        return "OFFICIAL"
    if any(x in domain for x in [".edu", "arxiv.org", "pubmed", "nature.com", "science.org"]):
        return "RESEARCH"
    if any(x in domain for x in ["github.com", "huggingface.co", "docs.python.org", "developer."]):
        return "TECHNICAL"
    if any(x in domain for x in ["reuters.com", "apnews.com", "bbc.com", "nytimes.com", "theguardian.com"]):
        return "SECONDARY"
    if any(x in domain for x in ["wikipedia.org", "britannica.com"]):
        return "PRIMARY"
    return "COMMENTARY"

def _entities(text):
    ents = set()
    for m in re.finditer(r"\b[A-Z][a-zA-Z0-9\-]{2,}\b", str(text or "")):
        ents.add(m.group(0).lower())
    for m in re.finditer(r"\b\d{4}\b", str(text or "")):
        ents.add(m.group(0))
    return ents


def _entity_alignment(claim, span):
    ce = _entities(claim)
    if not ce:
        return 1.0
    se = _entities(span)
    if not se:
        return 0.0
    return len(ce & se) / len(ce)


def _claim_years(text):
    return set(re.findall(r"\b(?:1[89]\d{2}|20\d{2})\b", str(text or "")))


def _date_alignment(claim, span):
    cy = _claim_years(claim)
    if not cy:
        return True
    return len(cy & _claim_years(span)) > 0


def _span_negates(claim, span):
    neg = {"not", "no", "never", "without", "fails", "failed", "denied",
           "denies", "lacks", "lack", "cannot", "false", "incorrect",
           "untrue", "disputed", "contradicts", "refutes"}
    span_low = str(span or "").lower()
    claim_kw = list(_tokens(claim))[:8]
    if not claim_kw:
        return False
    present_neg = set(re.findall(r"[a-z]+", span_low)) & neg
    if not present_neg:
        return False
    for neg_tok in present_neg:
        for m in re.finditer(r"\b" + re.escape(neg_tok) + r"\b", span_low):
            window = span_low[max(0, m.start() - 60):m.end() + 60]
            if any(kw in window for kw in claim_kw):
                return True
    return False


def _classify_entailment(claim, span, overlap, number_ok):
    if not span or overlap <= 0.0:
        return "UNSUPPORTED", 0.0
    entity_score = _entity_alignment(claim, span)
    date_ok = _date_alignment(claim, span)
    negated = _span_negates(claim, span)
    has_num = bool(_verification_numbers(claim))
    if negated and overlap >= 0.40:
        return "CONTRADICTORY", round(overlap, 3)
    if has_num and not number_ok and overlap >= 0.55:
        return "CONTRADICTORY", round(overlap, 3)
    if overlap >= 0.72 and entity_score >= 0.60 and number_ok and not negated and date_ok:
        return "CLEAR_SUPPORT", round(min(1.0, overlap * 0.55 + entity_score * 0.30 + 0.15), 3)
    if overlap >= 0.50 and entity_score >= 0.40 and number_ok and not negated:
        return "PARTIAL_SUPPORT", round(min(1.0, overlap * 0.65 + entity_score * 0.25), 3)
    if overlap >= 0.35:
        return "AMBIGUOUS", round(overlap, 3)
    return "UNSUPPORTED", round(overlap, 3)


_GROQ_ADJUDICATION_LOG = []


def _parse_groq_verdict(raw):
    s = str(raw or "").strip().upper()
    if not s:
        return None
    neg = ("NOT SUPPORT", "NO SUPPORT", "CANNOT SUPPORT", "DOES NOT SUPPORT",
           "DOESN'T SUPPORT", "DON'T SUPPORT", "NOT CONFIRM", "NOT VERIFIED",
           "NOT ENTAIL", "WITHOUT SUPPORT", "NOT CONTRADICT", "ISN'T CONTRADICT")
    has_neg = any(m in s for m in neg)
    if "CONTRADICT" in s:
        return "INSUFFICIENT" if has_neg else "CONTRADICTS"
    if has_neg:
        return "INSUFFICIENT"
    if any(w in s for w in ("INSUFFICIENT", "UNCLEAR", "AMBIGUOUS", "UNKNOWN", "INCONCLUSIVE", "CANNOT DETERMINE")):
        return "INSUFFICIENT"
    if any(w in s for w in ("SUPPORT", "CONFIRM", "VERIFIED", "ENTAIL", "TRUE", "CORRECT")):
        return "SUPPORTS"
    return None


def get_groq_adjudication_summary():
    if not _GROQ_ADJUDICATION_LOG:
        return ""
    lines = ["Groq adjudication: " + str(len(_GROQ_ADJUDICATION_LOG)) + " claim(s) independently re-checked."]
    for rec in _GROQ_ADJUDICATION_LOG[-10:]:
        lines.append("- verdict=" + str(rec.get("verdict")) + " | prior=" + str(rec.get("prior_status")) + " | claim=" + str(rec.get("claim", ""))[:90])
    return NL.join(lines)


async def selective_llm_verification(nodes, cfg, config):
    """Selective Groq entailment adjudication. Default-OFF. Never raises.
    C5 logs every adjudication. C6 strict verdict parsing. C7 high-risk priority."""
    try:
        if not getattr(cfg, "enable_llm_verification", False):
            return nodes
        max_checks = int(getattr(cfg, "max_llm_verifications", 3) or 3)
        fhr_raw = getattr(cfg, "force_adjudicate_high_risk", False)
        force_high_risk = fhr_raw.strip().lower() in {"1", "true", "yes", "on"} if isinstance(fhr_raw, str) else bool(fhr_raw)
        if max_checks <= 0 or not nodes:
            return nodes
        support_counts = {}
        for n in nodes:
            for tgt in (getattr(n, "supports", []) or []):
                support_counts[int(tgt)] = support_counts.get(int(tgt), 0) + 1
        high_risk_ids = set()
        for n in nodes:
            status = str(getattr(n, "verification_status", "") or "")
            ci = int(getattr(n, "citation_index", 0) or 0)
            is_contra = status == "CONTRADICTORY" or bool(getattr(n, "contradicts", []))
            is_hub = support_counts.get(ci, 0) >= 2
            if is_contra or is_hub:
                high_risk_ids.add(id(n))
        candidates = []
        for n in nodes:
            status = str(getattr(n, "verification_status", "") or "")
            claim = str(getattr(n, "claim", "") or "")
            span = str(getattr(n, "evidence_span", "") or "")
            if not claim or not span:
                continue
            is_ambiguous = status == "AMBIGUOUS"
            is_conflicting = status == "CONTRADICTORY" or bool(getattr(n, "contradicts", []))
            is_numeric = bool(_verification_numbers(claim))
            ent = float(getattr(n, "entailment_score", 0.0) or 0.0)
            is_high_risk = id(n) in high_risk_ids
            if is_ambiguous or is_conflicting or is_high_risk or (is_numeric and ent < 0.6):
                prio = (3.0 if is_high_risk else 0.0) + (2.0 if is_conflicting else 0.0) + (1.5 if is_ambiguous else 0.0) + (1.0 if is_numeric else 0.0) + (1.0 - ent)
                candidates.append((prio, is_high_risk, n))
        if not candidates:
            return nodes
        candidates.sort(key=lambda x: -x[0])
        hr_cap = max_checks + (2 if force_high_risk else 0)
        chosen = []
        for prio, is_hr, n in candidates:
            if len(chosen) >= hr_cap:
                break
            chosen.append(n)
        if not chosen:
            return nodes
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage
        for n in chosen:
            prior_status = str(getattr(n, "verification_status", "") or "")
            prior_ent = float(getattr(n, "entailment_score", 0.0) or 0.0)
            claim_short = str(getattr(n, "claim", ""))[:160]
            try:
                claim = str(getattr(n, "claim", ""))
                span = str(getattr(n, "evidence_span", ""))[:1500]
                prompt = ("Strict fact-check. Decide ONLY whether the EVIDENCE entails the CLAIM.\n"
                          "CLAIM: " + claim + "\nEVIDENCE: " + span + "\n"
                          "Reply with exactly one word: SUPPORTS, CONTRADICTS, or INSUFFICIENT.")
                model = init_chat_model(model=cfg.intake_model, max_tokens=16, api_key=get_api_key_for_model(cfg.intake_model, config))
                resp = await model.ainvoke([HumanMessage(content=prompt)])
                verdict = _parse_groq_verdict(getattr(resp, "content", ""))
                if verdict == "SUPPORTS":
                    n.entailment_score = max(prior_ent, 0.85)
                    if prior_status == "AMBIGUOUS":
                        n.verification_status = "PARTIAL_SUPPORT"
                elif verdict == "CONTRADICTS":
                    n.verification_status = "CONTRADICTORY"
                    n.entailment_score = min(prior_ent, 0.2)
                elif verdict == "INSUFFICIENT":
                    if prior_status in ("CLEAR_SUPPORT", "PARTIAL_SUPPORT"):
                        n.verification_status = "AMBIGUOUS"
                _GROQ_ADJUDICATION_LOG.append({"claim": claim_short, "verdict": verdict if verdict else "PARSE_FAILURE", "prior_status": prior_status, "new_status": str(getattr(n, "verification_status", "") or ""), "entailment_before": round(prior_ent, 3), "entailment_after": round(float(getattr(n, "entailment_score", 0.0) or 0.0), 3)})
            except Exception:
                _GROQ_ADJUDICATION_LOG.append({"claim": claim_short, "verdict": "ERROR", "prior_status": prior_status, "new_status": prior_status, "entailment_before": round(prior_ent, 3), "entailment_after": round(prior_ent, 3)})
                continue
        return nodes
    except Exception:
        return nodes



# ============================================================
# I15.6: STRONGER EVIDENCE ADJUDICATION
# High lexical overlap != automatic support.
# ============================================================
_I15_6_NEGATION_CUES = ("not ", "never ", "no longer", "without ", "failed to",
                        "did not", "didn't", "denied", "refuted", "contradicts", "disproves")
_I15_6_HEDGE_CUES = ("might", "could", "may ", "possibly", "perhaps", "potentially",
                     "expected to", "projected", "estimated", "likely", "unlikely")

def _i15_6_adjudicate_evidence(claim, span, context="", source_meta=None):
    """I15.6: Adjudicate claim vs span. Returns (verification_status, entailment_score).
    Checks negation, dates, units, qualifiers separately."""
    import re as _re
    claim = str(claim or "").strip()
    span = str(span or "").strip()
    if not claim or not span:
        return "UNSUPPORTED", 0.0
    claim_lower = claim.lower()
    span_lower = span.lower()
    claim_tokens = set(_re.findall(r"[a-z0-9]{2,}", claim_lower))
    span_tokens = set(_re.findall(r"[a-z0-9]{2,}", span_lower))
    if not claim_tokens:
        return "UNSUPPORTED", 0.0
    overlap = len(claim_tokens & span_tokens) / len(claim_tokens)
    claim_neg = any(c in claim_lower for c in _I15_6_NEGATION_CUES)
    span_neg = any(c in span_lower for c in _I15_6_NEGATION_CUES)
    negation_mismatch = claim_neg != span_neg
    claim_years = set(_re.findall(r"\b(19\d{2}|20\d{2})\b", claim))
    span_years = set(_re.findall(r"\b(19\d{2}|20\d{2})\b", span))
    date_conflict = bool(claim_years and span_years and not (claim_years & span_years))
    claim_nums = set(float(m) for m in _re.findall(r"\b(\d+(?:\.\d+)?)\b", claim))
    span_nums = set(float(m) for m in _re.findall(r"\b(\d+(?:\.\d+)?)\b", span))
    number_conflict = bool(claim_nums and span_nums and not (claim_nums & span_nums))
    claim_hedged = any(c in claim_lower for c in _I15_6_HEDGE_CUES)
    span_hedged = any(c in span_lower for c in _I15_6_HEDGE_CUES)
    qualifier_mismatch = claim_hedged != span_hedged
    if negation_mismatch and overlap >= 0.5:
        return "CONTRADICTORY", round(min(overlap, 0.4), 3)
    entailment = overlap
    if date_conflict:
        entailment *= 0.5
    if number_conflict:
        entailment *= 0.5
    if qualifier_mismatch:
        entailment *= 0.75
    if entailment >= 0.8 and not (date_conflict or number_conflict):
        status = "CLEAR_SUPPORT"
    elif entailment >= 0.5:
        status = "PARTIAL_SUPPORT"
    elif entailment >= 0.3:
        status = "AMBIGUOUS"
    else:
        status = "UNSUPPORTED"
    return status, round(entailment, 3)

async def verify_citations_programmatically(nodes):
    """I13.9: Bounded-concurrency programmatic citation verification.
    Per-URL timeout isolation, result caching, and input order preserved.
    Latency is bounded by batches, not N x timeout."""
    strong = []
    weak = []
    nodes = list(nodes or [])
    if not nodes:
        return {"strong": strong, "weak": weak}
    sem = asyncio.Semaphore(_VERIFY_CONCURRENCY)
    async def _fetch_text(client, url):
        response = await client.get(url)
        status_code = getattr(response, "status_code", 200)
        text = getattr(response, "text", "") or ""
        return status_code, text
    async def _verify_one(client, node):
        url = str(getattr(node, "url", "") or "").strip()
        claim = re.sub(r"\s+", " ", str(getattr(node, "claim", "") or "")).strip()
        if not url or not claim or not _safe_url(url):
            return None
        try:
            async with sem:
                status_code, page_text = await asyncio.wait_for(
                    _fetch_text(client, url), timeout=_VERIFY_URL_TIMEOUT)
        except Exception:
            return {"status": "UNSUPPORTED", "entailment": 0.0, "span": "", "provenance": "", "kind": "", "weak": True}
        if status_code >= 400:
            return {"status": "UNSUPPORTED", "entailment": 0.0, "span": "", "provenance": "", "kind": "", "weak": True}
        try:
            page = BeautifulSoup(page_text, "html.parser").get_text(" ", strip=True)
        except Exception:
            page = str(page_text)
        claim_tokens = _tokens(claim)
        best_span = ""
        best_score = 0.0
        for sentence in _sentence_chunks(page)[:400]:
            sentence_tokens = _tokens(sentence)
            if not claim_tokens or not sentence_tokens:
                continue
            overlap = len(claim_tokens.intersection(sentence_tokens)) / max(1, len(claim_tokens))
            if overlap > best_score:
                best_score = overlap
                best_span = sentence[:1800]
        if claim.lower() in page.lower():
            best_score = 1.0
            best_span = claim
        number_ok = _numbers_supported(claim, best_span)
        kind = _source_kind(url)
        status, ent_score = _classify_entailment(claim, best_span, best_score, number_ok)
        is_strong = status in ("CLEAR_SUPPORT", "PARTIAL_SUPPORT")
        span_out = best_span if is_strong else (best_span if status == "CONTRADICTORY" else "")
        prov = hashlib.sha256((url + "|" + claim + "|" + best_span).encode("utf-8")).hexdigest()[:16] if is_strong else ""
        return {"status": status, "entailment": ent_score, "span": span_out, "provenance": prov, "kind": kind, "weak": not is_strong}
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
        tasks = []
        for node in nodes[:40]:
            url = str(getattr(node, "url", "") or "").strip()
            claim = re.sub(r"\s+", " ", str(getattr(node, "claim", "") or "")).strip()
            if url and claim and _safe_url(url) and _verify_cache_get(url) is not None:
                tasks.append(None)
            else:
                tasks.append(_verify_one(client, node))
        results = await asyncio.gather(*[(t if t is not None else asyncio.sleep(0)) for t in tasks], return_exceptions=True)
    for idx, node in enumerate(nodes[:40]):
        url = str(getattr(node, "url", "") or "").strip()
        res = results[idx] if idx < len(results) else None
        if isinstance(res, Exception):
            res = None
        if res is None:
            res = _verify_cache_get(url)
        if res is None:
            res = {"status": "UNSUPPORTED", "entailment": 0.0, "span": "", "provenance": "", "kind": "", "weak": True}
        try:
            node.source_kind = res.get("kind", "") or ""
            node.verification_status = res.get("status", "UNSUPPORTED")
            node.entailment_score = float(res.get("entailment", 0.0) or 0.0)
            node.evidence_span = res.get("span", "") or ""
            node.provenance_id = res.get("provenance", "") or ""
            _domain = _domain_from_url(url)
            omega_memory.update_domain(_domain, not res.get("weak", True))
        except Exception:
            pass
        if url and _verify_cache_get(url) is None:
            _verify_cache_put(url, res)
        if res.get("weak", True):
            weak.append(node)
        else:
            strong.append(node)
    return {"strong": strong, "weak": weak}

def compute_research_frontier(evidence_graph, research_plan):
    """Classify each plan node by evidence coverage. Deterministic, zero tokens."""
    frontier = []
    evidence = evidence_graph or []
    plan = research_plan or []
    if not plan:
        return frontier
    for node in plan:
        if isinstance(node, dict):
            node_id = str(node.get("node_id", ""))
            topic = str(node.get("topic", ""))
        else:
            node_id = str(getattr(node, "node_id", ""))
            topic = str(getattr(node, "topic", ""))
        topic_tokens = _tokens(topic)
        if not topic_tokens:
            frontier.append({"node_id": node_id, "topic": topic, "status": "unanswered", "coverage": 0.0, "matched_count": 0, "has_contradiction": False})
            continue
        best_overlap = 0.0
        matched_contradictions = 0
        matched_count = 0
        for ev in evidence:
            claim = str(getattr(ev, "claim", "") or "")
            claim_tokens = _tokens(claim)
            if not claim_tokens:
                continue
            overlap = len(topic_tokens.intersection(claim_tokens)) / max(1, len(topic_tokens))
            if overlap >= 0.35:
                vstatus = str(getattr(ev, "verification_status", "UNVERIFIED") or "UNVERIFIED")
                if vstatus not in ("FAILED", "UNSUPPORTED", "CONTRADICTORY"):
                    matched_count += 1
                    if overlap > best_overlap:
                        best_overlap = overlap
                if getattr(ev, "contradicts", []):
                    matched_contradictions += 1
        if matched_count == 0:
            status = "unanswered"
        elif matched_contradictions > 0:
            status = "contradictory"
        elif best_overlap >= 0.7 and matched_count >= 2:
            status = "answered"
        else:
            status = "partially_answered"
        frontier.append({"node_id": node_id, "topic": topic, "status": status, "coverage": round(best_overlap, 3), "matched_count": matched_count, "has_contradiction": matched_contradictions > 0})
    return frontier


def compute_reasoning_depth(evidence_graph, research_plan):
    """Deterministic reasoning depth: novelty x contradiction x ambiguity x risk. Zero tokens."""
    evidence = evidence_graph or []
    frontier = compute_research_frontier(evidence_graph, research_plan)
    if not frontier:
        return {"depth_score": 0.0, "depth_tier": "minimal", "signals": {"novelty": 0.0, "contradiction_rate": 0.0, "ambiguity_rate": 0.0, "risk": 0.0}}
    total = len(frontier)
    unresolved = sum(1 for f in frontier if f["status"] in ("unanswered", "partially_answered"))
    contradictory = sum(1 for f in frontier if f["status"] == "contradictory")
    novelty = unresolved / total
    contra_nodes = sum(1 for ev in evidence if getattr(ev, "contradicts", []))
    contradiction_rate = contra_nodes / len(evidence) if evidence else 0.0
    amb_nodes = sum(1 for ev in evidence if str(getattr(ev, "verification_status", "") or "") == "AMBIGUOUS")
    ambiguity_rate = amb_nodes / len(evidence) if evidence else 0.0
    risk = min(1.0, (contradictory / total) * 2.0 + novelty * 0.5)
    depth_score = min(1.0, 0.35 * novelty + 0.25 * contradiction_rate + 0.20 * ambiguity_rate + 0.20 * risk)
    if depth_score >= 0.65:
        depth_tier = "deep"
    elif depth_score >= 0.40:
        depth_tier = "standard"
    else:
        depth_tier = "minimal"
    return {"depth_score": round(depth_score, 3), "depth_tier": depth_tier, "signals": {"novelty": round(novelty, 3), "contradiction_rate": round(contradiction_rate, 3), "ambiguity_rate": round(ambiguity_rate, 3), "risk": round(risk, 3)}}


def generate_frontier_branches(frontier, research_plan):
    """Turn unresolved frontier items into new DAG research nodes. Deterministic, zero tokens."""
    plan = list(research_plan) if research_plan else []
    existing_ids = set()
    existing_topics = set()
    for n in plan:
        if isinstance(n, dict):
            existing_ids.add(str(n.get("node_id", "")))
            existing_topics.add(str(n.get("topic", "")).strip().lower())
        else:
            existing_ids.add(str(getattr(n, "node_id", "")))
            existing_topics.add(str(getattr(n, "topic", "")).strip().lower())
    branches = []
    priority = {"contradictory": 0, "unanswered": 1, "partially_answered": 2}
    items = sorted(
        [f for f in (frontier or []) if f.get("status") in priority],
        key=lambda f: (priority.get(f.get("status"), 9), -float(f.get("coverage", 0.0)))
    )
    for item in items:
        if len(branches) >= 3:
            break
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        status = str(item.get("status", ""))
        if status == "contradictory":
            directive = "Resolve with independent sources this contradiction: " + topic
        elif status == "unanswered":
            directive = "Research this unresolved question: " + topic
        else:
            directive = "Deepen partial coverage of: " + topic
        if directive.strip().lower() in existing_topics:
            continue
        node_id = "FR_" + hashlib.sha256(directive.encode("utf-8")).hexdigest()[:6]
        if node_id in existing_ids:
            continue
        branches.append({"node_id": node_id, "topic": directive, "depends_on": []})
        existing_ids.add(node_id)
        existing_topics.add(directive.strip().lower())
    return branches


def generate_disconfirmation_branch(frontier, evidence_graph):
    """Generate a node that actively hunts counterevidence to the leading conclusion. Deterministic."""
    evidence = evidence_graph or []
    if len(evidence) < 2:
        return None
    best = None
    best_score = -1
    for n in evidence:
        claim = str(getattr(n, "claim", "")).strip()
        if not claim:
            continue
        score = len(getattr(n, "supports", []) or [])
        if score > best_score:
            best_score = score
            best = claim
    if not best:
        return None
    directive = ("Actively seek evidence that REFUTES this leading conclusion. "
                 "Search for counterexamples, retractions, competing measurements, and later corrections. "
                 "Conclusion to attack: " + best[:220])
    node_id = "DISCONF_" + hashlib.sha256(best.encode("utf-8")).hexdigest()[:6]
    return {"node_id": node_id, "topic": directive, "depends_on": []}


def compute_dynamic_search_budget(complexity_tier, base_budget, max_budget):
    """Scale researcher tool budget by question difficulty. Deterministic, zero tokens."""
    tier = str(complexity_tier or "Medium").strip()
    multipliers = {"Simple": 0.6, "Medium": 1.0, "Complex": 1.35, "Expert": 1.7}
    mult = multipliers.get(tier, 1.0)
    try:
        base = int(base_budget)
    except Exception:
        base = 6
    try:
        cap = int(max_budget)
    except Exception:
        cap = 10
    adjusted = int(round(base * mult))
    return max(3, min(cap, adjusted))


def calculate_epistemic_saturation(eg, rp):
    evidence = eg or []
    plan = rp or []
    if not plan: return min(1.0, len(evidence) / 20.0)
    covered = 0
    for node in plan:
        topic = node.get("topic", "") if isinstance(node, dict) else getattr(node, "topic", "")
        tt = _tokens(topic)
        if not tt:
            covered += 1
            continue
        best = 0.0
        for ev in evidence:
            ct = _tokens(getattr(ev, "claim", ""))
            if not ct: continue
            ov = len(tt & ct) / float(len(tt))
            if ov > best: best = ov
        if best >= 0.5: covered += 1
    base_cov = covered / float(len(plan))
    contradictions = sum(1 for n in evidence if getattr(n, "contradicts", []))
    evidence_bonus = min(0.15, len(evidence) / 50.0)
    penalty = min(0.15, contradictions * 0.03)
    return max(0.0, min(1.0, base_cov * 0.85 + evidence_bonus - penalty))


def _domain_authority(domain):
    """Return a conservative deterministic authority score for a domain.

    Scores are intentionally bounded and based only on the domain class
    already used by _source_kind().
    """
    d = str(domain or "").lower().strip()
    if not d:
        return 0.0

    # Highest authority: official/public-sector sources.
    if any(x in d for x in (".gov", ".mil")):
        return 1.0

    # Strong research / scientific sources.
    if any(
        x in d
        for x in (
            "nature.com",
            "science.org",
            "arxiv.org",
            "pubmed",
        )
    ) or ".edu" in d:
        return 0.95

    # Strong secondary/news sources.
    if any(
        x in d
        for x in (
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "nytimes.com",
            "theguardian.com",
        )
    ):
        return 0.90

    # Strong technical sources.
    if any(
        x in d
        for x in (
            "github.com",
            "huggingface.co",
            "docs.python.org",
            "developer.",
        )
    ):
        return 0.85

    # General reference sources.
    if any(
        x in d
        for x in (
            "wikipedia.org",
            "britannica.com",
        )
    ):
        return 0.75

    # Unknown / general domains remain usable but receive neutral weight.
    return 0.50

def programmatic_epistemic_verification(eg, ti):
    if not eg:
        return {"consensus_report": "No evidence.", "confidence_score": 0.0, "red_team_findings": "N/A", "devils_advocate_critique": "N/A"}
    unique = {}
    for n in eg:
        fp = " ".join(sorted(_tokens(getattr(n, "claim", ""))))
        if fp and fp not in unique:
            unique[fp] = n
    deduped = list(unique.values())
    year = datetime.now().year
    intent = str(ti or "").lower()
    valid = []
    decay = 0.0
    for n in deduped:
        score = 1.0
        ds = getattr(n, "date_published", None)
        if ds:
            try:
                age = year - int(str(ds)[:4])
                if intent == "current":
                    if age > 0:
                        score *= math.exp(-0.25 * age)
                elif intent == "predictive":
                    if age > 1:
                        score *= math.exp(-0.30 * (age - 1))
                decay += (1.0 - score)
            except Exception:
                pass
        if score > 0.35:
            valid.append((n, score))
    if not valid:
        return {"consensus_report": "No temporally-valid evidence survived filtering.", "confidence_score": 0.05, "red_team_findings": "Removed " + str(len(eg) - len(deduped)) + " duplicates; all evidence filtered by validity.", "devils_advocate_critique": "Evidence base empty after validity filtering."}
    domain_claims = {}
    for n, s in valid:
        d = _domain_from_url(getattr(n, "url", ""))
        domain_claims.setdefault(d, []).append(s)
    domains = [d for d in domain_claims if d]
    independent_sources = sum(math.sqrt(len(v)) for d, v in domain_claims.items() if d)
    source_independence = min(1.0, independent_sources / 4.0)
    auth_vals = [_domain_authority(d) for d in domains]
    source_quality = (sum(auth_vals) / len(auth_vals)) if auth_vals else 0.0
    trust_values = []
    for d in domains:
        trust_values.append(float(getattr(omega_memory, "data", {}).get("domains", {}).get(d, {}).get("trust", 0.5)))
    trust_score = sum(trust_values) / len(trust_values) if trust_values else 0.5
    candidate_contradictions = 0
    verified_contradictions = 0
    for n, _ in valid:
        if getattr(n, "contradicts", []):
            if str(getattr(n, "verification_status", "UNVERIFIED")) in ("VERIFIED", "CLEAR_SUPPORT", "PARTIAL_SUPPORT"):
                verified_contradictions += 1
            else:
                candidate_contradictions += 1
    contradictions = candidate_contradictions + verified_contradictions
    evidence_strength = 0.0
    for n, s in valid:
        d = _domain_from_url(getattr(n, "url", ""))
        evidence_strength += s * _domain_authority(d)
    evidence_strength = min(1.0, evidence_strength / 6.0)
    verification_ran = any(str(getattr(n, "verification_status", "UNVERIFIED")) != "UNVERIFIED" for n, _ in valid)
    if verification_ran:
        ent_scores = [float(getattr(n, "entailment_score", 0.0) or 0.0) for n, _ in valid]
        entailment = (sum(ent_scores) / len(ent_scores)) if ent_scores else 0.0
        verified_statuses = ("VERIFIED", "CLEAR_SUPPORT", "PARTIAL_SUPPORT")
        verified_count = sum(1 for n, _ in valid if str(getattr(n, "verification_status", "")) in verified_statuses)
        claim_coverage = verified_count / len(valid)
        ambiguous_count = sum(1 for n, _ in valid if str(getattr(n, "verification_status", "")) == "AMBIGUOUS")
        uncertainty = ambiguous_count / len(valid)
        base = claim_coverage * 0.20 + entailment * 0.25 + source_independence * 0.15 + source_quality * 0.20 + trust_score * 0.10 + evidence_strength * 0.10
        uncertainty_penalty = uncertainty * 0.15
    else:
        entailment = 0.0
        claim_coverage = 0.0
        uncertainty = 0.0
        base = source_independence * 0.25 + source_quality * 0.30 + evidence_strength * 0.25 + trust_score * 0.20
        uncertainty_penalty = 0.0
    temporal_fit = max(0.0, 1.0 - min(1.0, decay / max(1, len(valid))))
    base = base * (0.85 + 0.15 * temporal_fit)
    contradiction_penalty = verified_contradictions * 0.10 + candidate_contradictions * 0.04
    confidence = max(0.05, min(0.99, base - contradiction_penalty - uncertainty_penalty))
    has_authoritative = any(_domain_authority(d) >= 0.9 for d in domains)
    if not has_authoritative and confidence > 0.80:
        confidence = 0.80
    if contradictions > 0 and confidence > 0.70:
        confidence = 0.70
    if len(domains) < 2 and confidence > 0.60:
        confidence = 0.60
    if uncertainty > 0.5 and confidence > 0.65:
        confidence = 0.65
    consensus = "Analyzed " + str(len(valid)) + " facts from " + str(len(domains)) + " independent domains. Independence: " + str(round(source_independence, 2)) + ", quality: " + str(round(source_quality, 2)) + ". Confidence: " + str(round(confidence, 2))
    red_team = "Removed " + str(len(eg) - len(deduped)) + " duplicates. Contradictions: " + str(contradictions) + " (verified: " + str(verified_contradictions) + ")."
    critique = "Temporal fit: " + str(round(temporal_fit, 2)) + ". Verification-derived signals: " + ("active" if verification_ran else "not yet available") + "."
    return {"consensus_report": consensus, "confidence_score": confidence, "red_team_findings": red_team, "devils_advocate_critique": critique}



# ============================================================
# I14.9: REDIRECT-SAFETY BENCHMARK
# ============================================================
def _run_i14_9_redirect_safety_benchmark():
    """I14.9: Prove redirect-safe URL validation works correctly."""
    results = {"passed": 0, "failed": 0, "details": []}
    def check(name, condition):
        if condition: results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append("FAIL: " + name)
    # Test 1: Safe public URL passes
    safe1, reason1 = _i14_9_validate_url_deep("https://www.reuters.com/news/article")
    check("T1: public URL safe", safe1)
    # Test 2: Private IP rejected
    safe2, reason2 = _i14_9_validate_url_deep("http://192.168.1.1/admin")
    check("T2: private IP rejected", not safe2)
    # Test 3: Loopback rejected
    safe3, reason3 = _i14_9_validate_url_deep("http://127.0.0.1:8080/secret")
    check("T3: loopback rejected", not safe3)
    # Test 4: Cloud metadata rejected
    safe4, reason4 = _i14_9_validate_url_deep("http://169.254.169.254/latest/meta-data/")
    check("T4: metadata endpoint rejected", not safe4)
    # Test 5: Link-local rejected
    safe5, reason5 = _i14_9_validate_url_deep("http://169.254.1.1/internal")
    check("T5: link-local rejected", not safe5)
    # Test 6: localhost rejected
    safe6, reason6 = _i14_9_validate_url_deep("http://localhost:3000/api")
    check("T6: localhost rejected", not safe6)
    # Test 7: Non-HTTP scheme rejected
    safe7, reason7 = _i14_9_validate_url_deep("javascript:alert(1)")
    check("T7: javascript scheme rejected", not safe7)
    # Test 8: IP validation function
    ip_safe, _ = _i14_9_is_safe_ip("8.8.8.8")
    check("T8: public IP safe", ip_safe)
    ip_priv, _ = _i14_9_is_safe_ip("10.0.0.1")
    check("T8: private IP unsafe", not ip_priv)
    ip_meta, _ = _i14_9_is_safe_ip("169.254.169.254")
    check("T8: metadata IP unsafe", not ip_meta)
    # Test 9: Host validation
    host_safe, _ = _i14_9_validate_host("example.com")
    check("T9: public host safe", host_safe)
    host_local, _ = _i14_9_validate_host("localhost")
    check("T9: localhost host unsafe", not host_local)
    host_internal, _ = _i14_9_validate_host("metadata.google.internal")
    check("T9: internal host unsafe", not host_internal)
    # Test 10: Original _safe_url still works
    check("T10: _safe_url compatible", _safe_url("https://reuters.com/article"))
    check("T10: _safe_url blocks private", not _safe_url("http://192.168.1.1/x"))
    results["total"] = results["passed"] + results["failed"]
    results["success"] = results["failed"] == 0
    results["verdict"] = "PASS" if results["success"] else "FAIL"
    return results


# ============================================================
# I14.12: RESTRICTED EVALUATOR ABUSE BENCHMARK
# Zero tokens. Tests all attack vectors.
# ============================================================
def _run_i14_12_evaluator_benchmark():
    """I14.12: Prove restricted evaluator blocks all abuse patterns."""
    results = {"passed": 0, "failed": 0, "details": []}
    def check(name, condition):
        if condition: results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append("FAIL: " + name)
    r1 = python_repl.invoke({"code": "print(2 + 2)"})
    check("T1: normal computation works", "4" in r1)
    r2 = python_repl.invoke({"code": "x = ().__class__.__bases__[0].__subclasses__()"})
    check("T2: object traversal blocked", "BLOCKED" in r2 or "FALLBACK" in r2 or "Forbidden" in r2 or "RESTRICTED" in r2)
    r3 = python_repl.invoke({"code": "x = [0] * 100000000"})
    check("T3: giant allocation blocked", "BLOCKED" in r3 or "allocation" in r3.lower() or "RESTRICTED" in r3)
    r4 = python_repl.invoke({"code": "for i in range(100000000): pass"})
    check("T4: huge loop blocked", "BLOCKED" in r4 or "loop" in r4.lower() or "RESTRICTED" in r4)
    r5 = python_repl.invoke({"code": "print(\"x\" * 1000000)"})
    check("T5: output flooding blocked", "BLOCKED" in r5 or "flooding" in r5.lower() or "RESTRICTED" in r5)
    r6 = python_repl.invoke({"code": "import os"})
    check("T6: import blocked", "BLOCKED" in r6 or "forbidden" in r6.lower() or "RESTRICTED" in r6 or "Imports" in r6)
    r7 = python_repl.invoke({"code": "x = __import__(\"os\")"})
    check("T7: __import__ blocked", "BLOCKED" in r7 or "Forbidden" in r7 or "RESTRICTED" in r7)
    r8 = python_repl.invoke({"code": "exec(\"print(1)\")"})
    check("T8: exec blocked", "BLOCKED" in r8 or "Forbidden" in r8 or "RESTRICTED" in r8)
    r9 = python_repl.invoke({"code": "y = getattr(list, \"__name__\")"})
    check("T9: getattr blocked", "BLOCKED" in r9 or "reflection" in r9.lower() or "RESTRICTED" in r9)
    r10 = python_repl.invoke({"code": "import math; print(math.sqrt(144))"})
    check("T10: import blocked even for math", "BLOCKED" in r10 or "forbidden" in r10.lower() or "Imports" in r10 or "RESTRICTED" in r10)
    results["total"] = results["passed"] + results["failed"]
    results["success"] = results["failed"] == 0
    results["verdict"] = "PASS" if results["success"] else "FAIL"
    return results


# ============================================================
# I15.7: NATIVE TYPED TOOL RESULTS
# Every tool output is a ToolResult. FAILED never becomes evidence.
# ============================================================
_I15_7_FAILED_MARKERS = ("[FALLBACK]", "[JINA FALLBACK]", "[PDF] Download failed",
                         "[PDF] Unable to extract", "AUDIT_FAILED", "[PDF] Unsafe URL",
                         "[RESTRICTED EVALUATOR", "[TOOL_FAILED", "[TOOL_QUARANTINED")
_I15_7_DEGRADED_MARKERS = ("[TRUNCATED]", "[CHUNK", "No results", "No GitHub repos",
                           "No HF models", "No papers found", "No Wikipedia results",
                           "No Searxng results", "Jina failed", "Searxng failed",
                           "Wikipedia failed", "GitHub failed", "HF failed", "ArXiv failed",
                           "[TOOL_DEGRADED")

def _i15_7_classify_status(raw_output):
    """I15.7: Determine ToolResult status from raw output."""
    text = str(raw_output or "")
    if not text.strip():
        return "FAILED"
    if "[QUARANTINED" in text or "[TOOL_QUARANTINED" in text:
        return "QUARANTINED"
    for marker in _I15_7_FAILED_MARKERS:
        if marker in text:
            return "FAILED"
    for marker in _I15_7_DEGRADED_MARKERS:
        if marker in text:
            return "DEGRADED"
    return "SUCCESS"

# ============================================================
# I17.10: NATIVE ToolResult CONTRACT (canonical builder)
# Guaranteed-complete typed ToolResult. Status validated.
# ============================================================
_I17_10_TOOL_STATUSES = frozenset({"SUCCESS", "DEGRADED", "FAILED", "QUARANTINED"})

def _i17_10_validate_status(status):
    """I17.10: Coerce to a valid ToolResult status. Invalid -> FAILED."""
    s = str(status or "").strip().upper()
    if s in _I17_10_TOOL_STATUSES:
        return s
    return "FAILED"

def _i17_10_canonical_tool_result(status, source, content, error_class=None,
                                  source_result_id=None, final_url=None):
    """I17.10: Build a guaranteed-complete canonical ToolResult dict.
    All required fields always present; status validated against enum."""
    import time as _t
    import hashlib as _h
    now = _t.time()
    request_id = _h.sha256((str(source) + str(now)).encode("utf-8")).hexdigest()[:12]
    return {
        "status": _i17_10_validate_status(status),
        "source": str(source or "unknown"),
        "content": str(content or ""),
        "error_class": error_class,
        "request_id": request_id,
        "retrieved_at": now,
        "source_result_id": source_result_id,
        "final_url": final_url,
    }


# ============================================================
# I18.4: NATIVE ToolResult CONTRACT
# Tools return typed ToolResult. Status validated against enum.
# ============================================================
_I18_4_TOOL_STATUSES = frozenset({"SUCCESS", "DEGRADED", "FAILED", "QUARANTINED"})

def _i18_4_validate_status(status):
    """I18.4: Coerce to valid ToolResult status. Invalid -> FAILED."""
    s = str(status or "").strip().upper()
    if s in _I18_4_TOOL_STATUSES:
        return s
    return "FAILED"

def _i18_4_canonical_tool_result(status, source, content, error_class=None,
                                  source_result_id=None, final_url=None):
    """I18.4: Build guaranteed-complete canonical ToolResult dict."""
    import time as _t
    import hashlib as _h
    now = _t.time()
    request_id = _h.sha256((str(source) + str(now)).encode("utf-8")).hexdigest()[:12]
    return {
        "status": _i18_4_validate_status(status),
        "source": str(source or "unknown"),
        "content": str(content or ""),
        "error_class": error_class,
        "request_id": request_id,
        "retrieved_at": now,
        "source_result_id": source_result_id,
        "final_url": final_url,
    }


def _i15_7_make_tool_result(status, source, content, error_class=None, source_result_id=None, final_url=None):
    """I15.7: Construct a typed ToolResult."""
    import time as _t
    import hashlib as _h
    request_id = _h.sha256((str(source) + str(_t.time())).encode("utf-8")).hexdigest()[:12]
    return {
        "status": _i17_10_validate_status(status),
        "source": str(source or "unknown"),
        "content": str(content or ""),
        "error_class": error_class,
        "request_id": request_id,
        "retrieved_at": _t.time(),
        "source_result_id": source_result_id,
        "final_url": final_url,
    }

def _i15_7_to_tool_result(raw_output, tool_name="tool"):
    """I15.7: Normalize any raw tool output into a typed ToolResult."""
    status = _i15_7_classify_status(raw_output)
    return _i15_7_make_tool_result(status, tool_name, str(raw_output or ""))

def _i15_7_evidence_eligible(tool_result):
    """I15.7: Evidence-eligible only if SUCCESS or DEGRADED.
    FAILED and QUARANTINED can NEVER become evidence."""
    if not isinstance(tool_result, dict):
        return False
    return tool_result.get("status", "") in ("SUCCESS", "DEGRADED")

def _i15_7_filter_for_evidence(tool_results):
    """I15.7: Keep only evidence-eligible ToolResults."""
    return [tr for tr in (tool_results or []) if _i15_7_evidence_eligible(tr)]

def _run_i15_7_typed_tool_benchmark():
    """I15.7: Prove FAILED tool calls never become evidence."""
    results = {"passed": 0, "failed": 0, "details": []}
    def check(name, condition):
        if condition: results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append("FAIL: " + name)
    failed = _i15_7_to_tool_result("[FALLBACK] Tool failed.", "search")
    check("T1: FAILED classified", failed["status"] == "FAILED")
    check("T1: has request_id", bool(failed["request_id"]))
    check("T1: has retrieved_at", failed["retrieved_at"] > 0)
    check("T1: FAILED not evidence-eligible", not _i15_7_evidence_eligible(failed))
    success = _i15_7_to_tool_result("IBM announced a 1000-qubit processor. https://reuters.com/x", "search")
    check("T2: SUCCESS classified", success["status"] == "SUCCESS")
    check("T2: SUCCESS evidence-eligible", _i15_7_evidence_eligible(success))
    quar = _i15_7_to_tool_result("[QUARANTINED score=5] malicious payload", "evil")
    check("T3: QUARANTINED classified", quar["status"] == "QUARANTINED")
    check("T3: QUARANTINED not eligible", not _i15_7_evidence_eligible(quar))
    deg = _i15_7_to_tool_result("No results found for query", "search")
    check("T4: DEGRADED classified", deg["status"] == "DEGRADED")
    check("T4: DEGRADED eligible", _i15_7_evidence_eligible(deg))
    empty = _i15_7_to_tool_result("", "search")
    check("T5: empty is FAILED", empty["status"] == "FAILED")
    batch = [failed, success, quar, deg]
    eligible = _i15_7_filter_for_evidence(batch)
    check("T6: filter keeps 2 of 4", len(eligible) == 2)
    check("T6: no FAILED in eligible", all(tr["status"] != "FAILED" for tr in eligible))
    check("T6: no QUARANTINED in eligible", all(tr["status"] != "QUARANTINED" for tr in eligible))
    check("T7: FAILED blocked from evidence", not _i15_7_evidence_eligible(failed))
    results["total"] = results["passed"] + results["failed"]
    results["success"] = results["failed"] == 0
    results["verdict"] = "PASS" if results["success"] else "FAIL"
    return results

async def get_search_tool(sa):
    if sa == SearchAPI.SEARXNG: return [searxng_search]
    if sa == SearchAPI.NONE: return []
    return [jina_search]


async def get_all_tools(config):
    cfg = Configuration.from_runnable_config(config)
    tools = [research_complete_tool, think_tool, omega_pdf_ingestor, github_sniper, huggingface_sniper, wikipedia_rest_search, arxiv_search, audit_pricing]
    if getattr(cfg, "enable_python_repl", True): tools.append(python_repl)
    tools.extend(await get_search_tool(SearchAPI(get_config_value(cfg.search_api))))
    return tools

def get_notes_from_tool_calls(msgs):
    try:
        return [str(m.content) for m in filter_messages(msgs, include_types=["tool"])]
    except Exception:
        return [str(getattr(m, "content", "")) for m in msgs if getattr(m, "type", "") == "tool"]

def is_token_limit_exceeded(e, model=None):
    s = str(e).lower()
    return "context_length" in s or "too long" in s or "resource_exhausted" in s or "413" in s or "maximum context" in s

def get_model_token_limit(m):
    m = str(m or "").lower()
    if "groq:llama-3.3-70b" in m: return 128000
    if "groq:llama-3.1-8b" in m: return 128000
    return None

def check_information_satiation(new, existing, threshold=0.75):
    if not new: return True
    new_text = " ".join([str(x) for x in new])
    old_text = " ".join([str(x) for x in (existing or [])])
    nt = _tokens(new_text)
    if not nt: return False
    ot = _tokens(old_text)
    if not ot: return False
    return len(nt.intersection(ot)) / len(nt) >= float(threshold)

def filter_and_verify_evidence(eg, temporal_intent="Current"):
    """Normalize, validate, deduplicate and cap evidence deterministically."""
    out = []
    seen = set()

    for node in eg or []:
        claim = str(getattr(node, "claim", "")).strip()
        url = str(getattr(node, "url", "")).strip()

        if len(claim) < 20 or not _safe_url(url):
            continue

        key = (
            re.sub(r"\s+", " ", claim.lower()),
            url.rstrip("/").lower()
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(node)

    return out[:80]

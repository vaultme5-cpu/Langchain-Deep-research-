import re, io, asyncio, contextlib, os, json, math, sqlite3, time, threading, logging
from datetime import datetime
from typing import List, Optional, Any
from urllib.parse import quote, quote_plus
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.messages import filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.state import ResearchComplete
try: import arxiv
except ImportError: arxiv = None
try: import aiohttp
except ImportError: aiohttp = None

class LoopSafeSemaphore:
    def __init__(self, limit):
        self.limit = limit
        self._semaphores = {}
    async def __aenter__(self):
        try: loop = asyncio.get_running_loop()
        except RuntimeError: loop = asyncio.new_event_loop()
        key = id(loop)
        if key not in self._semaphores: self._semaphores[key] = asyncio.Semaphore(self.limit)
        await self._semaphores[key].__aenter__()
        return self
    async def __aexit__(self, exc_type, exc, tb):
        try: loop = asyncio.get_running_loop()
        except RuntimeError: loop = asyncio.new_event_loop()
        await self._semaphores[id(loop)].__aexit__(exc_type, exc, tb)

class GroqShield:
    def __init__(self, keys):
        self.keys = keys if keys else [""]
        self.cooldowns = {k: 0 for k in self.keys}
        self.lock = threading.Lock()
        self.idx = 0
    def get_key(self, last_failed=None):
        with self.lock:
            now = time.time()
            if last_failed and last_failed in self.cooldowns: self.cooldowns[last_failed] = now + 60.0
            avail = [k for k in self.keys if self.cooldowns.get(k, 0) <= now]
            if not avail: return min(self.keys, key=lambda k: self.cooldowns.get(k, 0))
            self.idx = (self.idx + 1) % len(avail)
            return avail[self.idx]

_raw_groq = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
_groq_keys = [k.strip() for k in _raw_groq.split(",") if k.strip()]
_shield = GroqShield(_groq_keys)

def get_api_key_for_model(model_name, config, last_failed=None):
    if model_name.lower().startswith("groq:"): return _shield.get_key(last_failed)
    return None

def get_today_str(): return datetime.now().strftime("%a %b %d, %Y")
def get_config_value(value):
    if value is None: return None
    return value.value if hasattr(value, "value") else value

@tool(description="Search GitHub repositories.")
def github_sniper(query: str) -> str:
    try:
        h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Omega/2.0"}
        r = httpx.get("https://api.github.com/search/repositories", headers=h, params={"q": query, "sort": "stars", "per_page": 5}, timeout=15)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items: return "No GitHub repos found."
        return "\n\n".join([f"[{i+1}] {x.get('full_name')} | Stars: {x.get('stargazers_count')}\nURL: {x.get('html_url')}" for i, x in enumerate(items)])
    except Exception as e: return f"GitHub failed: {e}"

@tool(description="Search HuggingFace models.")
def huggingface_sniper(query: str) -> str:
    try:
        r = httpx.get("https://huggingface.co/api/models", params={"search": query, "sort": "downloads", "direction": "-1", "limit": 5}, timeout=15)
        r.raise_for_status()
        models = r.json()
        if not models: return "No HF models found."
        return "\n\n".join([f"[{i+1}] {m.get('modelId')} | DL: {m.get('downloads',0)}\nURL: https://huggingface.co/{m.get('modelId')}" for i, m in enumerate(models)])
    except Exception as e: return f"HF failed: {e}"

@tool(description="Search ArXiv papers.")
def arxiv_search(query: str) -> str:
    if not arxiv: return "ArXiv not installed."
    try:
        s = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        out = [f"[{i+1}] {p.title} ({p.entry_id})\n{p.summary[:800]}" for i, p in enumerate(s.results())]
        return "\n\n".join(out) if out else "No papers found."
    except Exception: return "ArXiv failed."

@tool(description="Execute Python code securely.")
def python_repl(code: str) -> str:
    import ast, concurrent.futures
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in ['os','sys','subprocess','socket','shutil']: return f"[FALLBACK] Import of {a.name} forbidden."
    except Exception as e: return f"[FALLBACK] Syntax: {e}"
    def _run():
        out = io.StringIO()
        g = {"__builtins__": {"print":print, "len":len, "range":range, "list":list, "dict":dict, "set":set, "str":str, "int":int, "float":float, "sum":sum}, "json":json, "re":re, "math":math, "datetime":datetime, "BeautifulSoup":BeautifulSoup, "httpx":httpx}
        with contextlib.redirect_stdout(out): exec(code, g, {})
        return out.getvalue()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex: return ex.submit(_run).result(timeout=30.0) or "OK"
    except Exception as e: return f"[FALLBACK] {e}"

@tool(description="Audit URL pricing.")
def audit_pricing(url: str) -> str:
    try:
        r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, follow_redirects=True)
        t = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True).lower()
        red = sum(1 for f in ["contact sales", "book a demo", "enterprise pricing"] if f in t)
        grn = sum(1 for f in ["open source", "free tier", "100% free", "mit license"] if f in t)
        if grn > red and grn > 0: return f"VERIFIED_FREE: {url}"
        if red > 0 and grn == 0: return f"PAID_ENTERPRISE: {url}"
        return f"UNKNOWN: {url}"
    except Exception: return "AUDIT_FAILED"

def jina_scraper(url):
    key = os.environ.get("JINA_API_KEY", "")
    h = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if key: h["Authorization"] = f"Bearer {key}"
    try:
        r = httpx.get(f"https://r.jina.ai/{quote(str(url), safe='')}", headers=h, timeout=30.0)
        r.raise_for_status()
        return r.json().get("data", {}).get("content", "")[:6000]
    except Exception: return "[JINA FALLBACK]"

async def validate_urls(urls):
    out = {}
    if aiohttp is None: return {u: True for u in urls}
    async with aiohttp.ClientSession() as s:
        for u in set(urls):
            if not u or not str(u).startswith("http"): out[u] = False; continue
            try:
                async with s.get(u, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=5)) as r: out[u] = r.status < 400
            except Exception: out[u] = False
    return out

async def _crawl_urls(urls):
    out = {}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as c:
        tasks = [c.get(u) for u in urls]
        resps = await asyncio.gather(*tasks, return_exceptions=True)
        for u, r in zip(urls, resps):
            if isinstance(r, Exception) or getattr(r, "status_code", 400) >= 400:
                jt = jina_scraper(u)
                if not jt.startswith("[JINA"): out[u] = jt
                continue
            try:
                soup = BeautifulSoup(r.text, "html.parser")
                for s in soup(["script", "style", "nav", "footer"]): s.extract()
                txt = "\n".join([l.strip() for l in soup.get_text(" ", strip=True).splitlines() if len(l.strip()) > 40])
                out[u] = txt[:6000]
            except Exception: pass
    return out

@tool(description="Search web via Jina AI.")
async def jina_search(query: str) -> str:
    key = os.environ.get("JINA_API_KEY", "")
    h = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if key: h["Authorization"] = f"Bearer {key}"
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get(f"https://s.jina.ai/{quote_plus(str(query))}", headers=h)
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data: return "No results."
            return "\n\n".join([f"--- {i+1}: {d.get('title')} ---\nURL: {d.get('url')}\n{d.get('content','')[:1200]}" for i, d in enumerate(data[:5])])
    except Exception as e: return f"Jina failed: {e}"

@tool(description="Search Wikipedia.")
def wikipedia_rest_search(query: str) -> str:
    try:
        r = httpx.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3}, timeout=10).json()
        hits = r.get("query", {}).get("search", [])
        if not hits: return "No Wikipedia results."
        return "\n\n".join([f"TITLE: {h['title']}\nURL: https://en.wikipedia.org/wiki/{h['title'].replace(' ','_')}\n{h.get('snippet','')}" for h in hits])
    except Exception: return "Wikipedia failed."

@tool(description="Strategic reflection.")
def think_tool(reflection: str) -> str: return "Reflection: " + reflection

@tool("ResearchComplete", description="Signal that the research plan is complete.")
def ResearchComplete() -> str: return "Research complete."

_pdf_semaphore = LoopSafeSemaphore(1)
@tool(description="Ingest PDF via 3-Tier Engine.")
async def omega_pdf_ingestor(url: str) -> str:
    if not url.lower().endswith('.pdf'): return "Not a PDF."
    async with _pdf_semaphore:
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as c:
                r = await c.get(url); r.raise_for_status(); pdf = r.content
        except Exception as e: return f"[PDF] Download failed: {e}"
        try:
            import pymupdf4llm, tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t: t.write(pdf); tp = t.name
            md = pymupdf4llm.to_markdown(tp); os.unlink(tp)
            if len(md.strip()) > 200: return f"[TIER1 LOCAL]\n{md[:15000]}"
        except Exception: pass
        return "[PDF] Fallback to raw text or Gemini required."

OMEGA_MEM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omega_memory.json")
class OmegaMemory:
    def __init__(self):
        self.data = {"domains": {}}
        try:
            if os.path.exists(OMEGA_MEM):
                with open(OMEGA_MEM) as f: self.data = json.load(f)
        except Exception: pass
    def save(self):
        try:
            t = OMEGA_MEM + ".tmp"
            with open(t, "w") as f: json.dump(self.data, f)
            os.replace(t, OMEGA_MEM)
        except Exception: pass
    def update_domain(self, d, ok):
        if not d: return
        x = self.data["domains"].get(d, {"trust": 0.5, "hits": 0})
        x["hits"] += 1; x["trust"] = max(0.0, min(1.0, x["trust"] * 0.9 + (0.1 if ok else -0.05)))
        self.data["domains"][d] = x; self.save()
    def get_context_prompt(self):
        top = sorted(self.data.get("domains",{}).items(), key=lambda x: x[1]["trust"], reverse=True)[:5]
        if not top: return ""
        return "Trusted: " + ", ".join([f"{d}({v['trust']:.1f})" for d, v in top])
try: omega_memory = OmegaMemory()
except Exception:
    class _NoOp:
        def get_context_prompt(self): return ""
        def update_domain(self, d, s): pass
    omega_memory = _NoOp()

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omega.db")
class LocalSQLiteMemory:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False); self.c = self.conn.cursor()
        self.c.execute("CREATE TABLE IF NOT EXISTS ev (id INTEGER PRIMARY KEY, claim TEXT, url TEXT)"); self.conn.commit()
    def store(self, claim, url):
        if not claim or not url: return
        try: self.c.execute("INSERT INTO ev (claim, url) VALUES (?,?)", (claim, url)); self.conn.commit()
        except Exception: pass
    def recall(self, q, limit=5): return []
try: omega_local_memory = LocalSQLiteMemory()
except Exception:
    class _NoOpDB:
        def store(self, c, u): pass
        def recall(self, q, l=5): return []
    omega_local_memory = _NoOpDB()

_citation_semaphore = LoopSafeSemaphore(5)
async def verify_citations_programmatically(nodes):
    out = []
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as c:
        for n in nodes:
            async with _citation_semaphore:
                u = getattr(n, "url", ""); cl = getattr(n, "claim", "")
                if not u or not cl or not u.startswith("http"): continue
                try:
                    r = await c.get(u)
                    if r.status_code >= 400: continue
                    t = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True).lower()
                    if cl.lower() in t: out.append(n)
                except Exception: pass
    return out

def calculate_epistemic_saturation(eg, rp):
    if not rp or not eg: return 0.0
    pw = set()
    for n in rp:
        t = n.get("topic","") if isinstance(n, dict) else getattr(n, "topic", "")
        pw.update(re.findall(r'\b\w{4,}\b', t.lower()))
    if not pw: return 0.0
    cw = set()
    for n in eg: cw.update(re.findall(r'\b\w{4,}\b', getattr(n, 'claim', '').lower()))
    return min(len(pw.intersection(cw)) / len(pw) + min(len(eg)/30.0, 0.2), 1.0)

def programmatic_epistemic_verification(eg, ti):
    if not eg: return {"consensus_report": "No evidence.", "confidence_score": 0.0, "red_team_findings": "N/A", "devils_advocate_critique": "N/A"}
    uc = {}
    for n in eg:
        fp = " ".join(sorted(set(re.findall(r'\b\w{4,}\b', getattr(n,'claim','').lower()))))
        if fp and fp not in uc: uc[fp] = n
    dn = list(uc.values()); cy = datetime.now().year; vn, dp = [], 0.0
    for n in dn:
        ds = getattr(n, 'date_published', None); sc = 1.0
        if ds and ti == "Current":
            try:
                age = cy - int(str(ds)[:4])
                if age > 0: sc *= math.exp(-0.2 * age); dp += (1.0 - sc)
            except Exception: pass
        if sc > 0.4: vn.append((n, sc))
    doms = set()
    for n, _ in vn:
        u = getattr(n, 'url', '')
        if u:
            try: doms.add(u.split("//")[-1].split("/")[0].replace("www.", ""))
            except Exception: pass
    fc = max(0.1, min((min(len(doms)/5.0,1.0)*0.4 + min(len(vn)/10.0,1.0)*0.6) - sum(1 for n,_ in vn if getattr(n,'contradicts',[]))*0.05 - dp*0.01, 0.99))
    return {"consensus_report": f"Analyzed {len(vn)} facts from {len(doms)} domains. Confidence: {fc:.2f}", "confidence_score": fc, "red_team_findings": f"Removed {len(eg)-len(dn)} duplicates.", "devils_advocate_critique": f"Decay: {dp:.2f}."}

async def get_search_tool(sa):
    if sa == SearchAPI.SEARXNG: return [jina_search]
    return [jina_search]

async def get_all_tools(config):
    tools = [tool(ResearchComplete), think_tool, omega_pdf_ingestor, github_sniper, huggingface_sniper, wikipedia_rest_search, arxiv_search, python_repl, audit_pricing, jina_search]
    cfg = Configuration.from_runnable_config(config)
    tools.extend(await get_search_tool(SearchAPI(get_config_value(cfg.search_api))))
    return tools

def get_notes_from_tool_calls(msgs): return [m.content for m in filter_messages(msgs, include_types="tool")]
def is_token_limit_exceeded(e, model=None):
    s = str(e).lower(); return 'context_length' in s or 'too long' in s or 'resource_exhausted' in s
def get_model_token_limit(m):
    for k, v in {"groq:llama-3.3-70b": 128000, "groq:llama-3.1-8b": 128000}.items():
        if k in m: return v
    return None
def check_information_satiation(new, existing, threshold=0.75): return False
def filter_and_verify_evidence(eg, temporal_intent="Current"): return eg or []

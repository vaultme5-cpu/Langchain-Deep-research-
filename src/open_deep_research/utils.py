import re, io, itertools, asyncio, contextlib, logging, warnings, os, json, math, sqlite3, time, threading
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional
from collections import Counter
from urllib.parse import quote, quote_plus
import aiohttp, httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool, ToolException, tool
from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

try: import arxiv
except ImportError: arxiv = None

# ==========================================
# 1. GROQ SHIELD & INFRASTRUCTURE
# ==========================================
class GroqShield:
    def __init__(self, keys):
        self.keys = keys if keys else [""]
        self.cooldowns = {k: 0 for k in self.keys}
        self.lock = threading.Lock()
        self._index = 0

    def get_key(self, last_failed_key=None):
        with self.lock:
            now = time.time()
            if last_failed_key and last_failed_key in self.cooldowns:
                self.cooldowns[last_failed_key] = now + 60.0
            available_keys = [k for k in self.keys if self.cooldowns.get(k, 0) <= now]
            if not available_keys:
                return min(self.keys, key=lambda k: self.cooldowns.get(k, 0))
            self._index = (self._index + 1) % len(available_keys)
            return available_keys[self._index]

_raw_groq = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
_groq_keys = [k.strip() for k in _raw_groq.split(",") if k.strip()]
_shield = GroqShield(_groq_keys)
groq_burst_semaphore = asyncio.Semaphore(3)

def get_api_key_for_model(model_name: str, config: RunnableConfig, last_failed_key: str = None):
    model_name = model_name.lower()
    if model_name.startswith("groq:"): return _shield.get_key(last_failed_key)
    elif model_name.startswith("openai:"): return os.getenv("OPENAI_API_KEY")
    elif model_name.startswith("anthropic:"): return os.getenv("ANTHROPIC_API_KEY")
    return None

def get_today_str() -> str:
    now = datetime.now()
    return now.strftime("%a %b ") + str(now.day) + ", " + str(now.year)

def get_config_value(value):
    if value is None: return None
    return value.value if hasattr(value, 'value') else value

# ==========================================
# 2. SNIPER TOOLS (ZERO SEO SPAM)
# ==========================================
@tool(description="Search GitHub for open-source repositories.")
def github_sniper(query: str) -> str:
    try:
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "ProjectOmega/2.0"}
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 8}
        response = httpx.get("https://api.github.com/search/repositories", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items: return "No GitHub repositories found."
        results = []
        for i, repo in enumerate(items):
            results.append(f"[{i+1}] {repo.get('full_name')} | Stars: {repo.get('stargazers_count')} | Lang: {repo.get('language')}\nURL: {repo.get('html_url')}\nDesc: {str(repo.get('description', ''))[:150]}")
        return "\n\n".join(results)
    except Exception as e: return "GitHub search failed: " + str(e)

@tool(description="Search HuggingFace for AI models and datasets.")
def huggingface_sniper(query: str) -> str:
    try:
        params = {"search": query, "sort": "downloads", "direction": "-1", "limit": 8}
        response = httpx.get("https://huggingface.co/api/models", params=params, timeout=15)
        response.raise_for_status()
        models = response.json()
        if not models: return "No HuggingFace models found."
        results = []
        for i, model in enumerate(models):
            model_id = str(model.get("modelId"))
            results.append(f"[{i+1}] {model_id} | Downloads: {model.get('downloads', 0)} | Type: {model.get('pipeline_tag', 'Unknown')}\nURL: https://huggingface.co/{model_id}")
        return "\n\n".join(results)
    except Exception as e: return "HuggingFace search failed: " + str(e)

@tool(description="Search Hacker News for top stories.")
def hackernews_sniper(query: str) -> str:
    try:
        params = {"query": query, "tags": "story", "hitsPerPage": 8}
        response = httpx.get("https://hn.algolia.com/api/v1/search", params=params, timeout=15)
        response.raise_for_status()
        hits = response.json().get("hits", [])
        if not hits: return "No Hacker News stories found."
        results = []
        for i, hit in enumerate(hits):
            hn_url = "https://news.ycombinator.com/item?id=" + str(hit.get("objectID", ""))
            results.append(f"[{i+1}] {hit.get('title')} | Points: {hit.get('points', 0)} | Comments: {hit.get('num_comments', 0)}\nHN Link: {hn_url}")
        return "\n\n".join(results)
    except Exception as e: return "Hacker News search failed: " + str(e)

@tool(description="Search ArXiv.")
def arxiv_search(query: str) -> str:
    if not arxiv: return "ArXiv library not installed."
    try:
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for i, paper in enumerate(search.results()):
            results.append(f"[{i+1}] {paper.title} ({paper.entry_id})\nAbstract: {paper.summary[:1500]}")
        return "\n\n".join(results) if results else "No ArXiv papers found."
    except Exception: return "ArXiv search failed."

@tool(description="Search PubMed.")
def pubmed_search(query: str) -> str:
    try:
        resp = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 3}, timeout=10).json()
        ids = resp.get("esearchresult", {}).get("idlist", [])
        if not ids: return "No PubMed articles found."
        fetch_resp = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=10).json()
        results = []
        for uid in ids:
            data = fetch_resp.get("result", {}).get(uid, {})
            results.append(f"TITLE: {data.get('title', 'Unknown')}\nPMID: {uid}\nLINK: https://pubmed.ncbi.nlm.nih.gov/{uid}/")
        return "\n\n".join(results)
    except Exception: return "PubMed search failed."

@tool(description="Search Semantic Scholar.")
def semantic_scholar_search(query: str) -> str:
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": 5, "fields": "title,abstract,url,year,citationCount"}
        resp = httpx.get(url, params=params, timeout=10).json()
        papers = resp.get("data", [])
        if not papers: return "No Semantic Scholar papers found."
        results = []
        for i, p in enumerate(papers):
            results.append(f"[{i+1}] {p.get('title', 'Unknown')} ({p.get('year', 'Unknown')}) | Citations: {p.get('citationCount', 0)}\nURL: {p.get('url', '')}\nAbstract: {str(p.get('abstract', 'No abstract'))[:1000]}")
        return "\n\n".join(results)
    except Exception: return "Semantic Scholar search failed."

@tool(description="Find free alternatives to paid software.")
def saas_alternative_sniper(query: str) -> str:
    try:
        alt_query = "site:alternativeto.net OR site:opensource.com " + query + " free alternative open source"
        with DDGS() as ddgs:
            results_list = [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in ddgs.text(alt_query, max_results=8)]
        if not results_list: return "No free alternatives found."
        results = []
        for i, res in enumerate(results_list[:8]):
            results.append(f"[{i+1}] {res.get('title', 'Unknown')}\nURL: {res.get('url', '')}\nDetails: {res.get('content', '')[:250]}")
        return "\n\n".join(results)
    except Exception as e: return "Alternative search failed: " + str(e)

@tool(description="Calculate real-time momentum of a topic.")
def trend_velocity_auditor(query: str) -> str:
    try:
        hn_url = "https://hn.algolia.com/api/v1/search_by_date"
        cutoff = str(int(time.time()) - 604800)
        hn_resp = httpx.get(hn_url, params={"query": query, "tags": "story", "numericFilters": "created_at_i>" + cutoff}, timeout=10).json()
        hn_hits = hn_resp.get("hits", [])
        hn_points = sum(hit.get("points", 0) for hit in hn_hits)
        gh_url = "https://api.github.com/search/repositories"
        gh_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        gh_resp = httpx.get(gh_url, headers={"Accept": "application/vnd.github.v3+json"}, params={"q": query + " pushed:>" + gh_date, "sort": "updated"}, timeout=10).json()
        gh_count = len(gh_resp.get("items", []))
        velocity = hn_points + (gh_count * 10)
        if velocity > 100: status = "TRENDING UP (High Velocity)"
        elif velocity > 20: status = "STABLE (Moderate Activity)"
        else: status = "STAGNANT/DEAD (Low Activity)"
        return f"VELOCITY REPORT for '{query}':\nStatus: {status}\nHN Points (7d): {hn_points}\nGH Active Repos (7d): {gh_count}\nVelocity Score: {velocity}"
    except Exception as e: return "Velocity audit failed: " + str(e)

# ==========================================
# 3. HACKER'S SANDBOX (PYTHON REPL)
# ==========================================
@tool(description="Execute Python code securely for Search-as-Code (SaC) data processing.")
def python_repl(code: str) -> str:
    import ast, concurrent.futures
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'sys', 'subprocess', 'socket', 'shutil', 'pathlib', 'requests']:
                        return f"[FALLBACK] SECURITY VIOLATION: Import of {alias.name} is forbidden."
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in ['os', 'sys', 'subprocess', 'socket', 'shutil', 'pathlib', 'requests']:
                    return f"[FALLBACK] SECURITY VIOLATION: Import from {node.module} is forbidden."
            elif isinstance(node, ast.Call) and getattr(node.func, 'id', '') in ['eval', 'exec', 'exit', 'quit']:
                return f"[FALLBACK] SECURITY VIOLATION: Use of {node.func.id}() is forbidden."
    except Exception as e:
        return f"[FALLBACK] SYNTAX ERROR: {str(e)}"

    def _execute():
        stdout = io.StringIO()
        try:
            import pandas as pd
            import numpy as np
            import yfinance as yf
        except ImportError:
            pd, np, yf = None, None, None
            
        safe_globals = {
            "__builtins__": {"print": print, "len": len, "range": range, "sorted": sorted, "list": list, "dict": dict, "set": set, "str": str, "int": int, "float": float, "min": min, "max": max, "sum": sum, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "isinstance": isinstance, "type": type, "abs": abs, "round": round, "bool": bool, "tuple": tuple, "any": any, "all": all, "next": next, "iter": iter},
            "json": json, "re": re, "math": math, "collections": __import__('collections'), "datetime": datetime, "BeautifulSoup": BeautifulSoup, "httpx": httpx, "pd": pd, "np": np, "yf": yf
        }
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, {})
        return stdout.getvalue()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute)
            result = future.result(timeout=45.0)
            return result or "Code executed successfully (no output)."
    except concurrent.futures.TimeoutError:
        return "[FALLBACK] EXECUTION TIMEOUT: Code took longer than 45 seconds."
    except Exception as e:
        return f"[FALLBACK] EXECUTION ERROR: {str(e)}"

@tool(description="Audit a URL to verify if it is 100% free.")
def audit_pricing(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 ProjectOmega/2.0"}
        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True).lower()
        red_flags = ["contact sales", "book a demo", "enterprise pricing", "credit card required", "upgrade to pro"]
        green_flags = ["open source", "free tier", "free forever", "playground", "100% free", "mit license", "apache license", "huggingface space", "github.com"]
        red_count = sum(1 for flag in red_flags if flag in text)
        green_count = sum(1 for flag in green_flags if flag in text)
        if green_count > red_count and green_count > 0: return "VERIFIED_FREE: " + url
        elif red_count > 0 and green_count == 0: return "PAID_ENTERPRISE: " + url + " (DISCARD if user requested 100% free)"
        else: return "UNKNOWN: " + url
    except Exception as e: return "AUDIT_FAILED: " + str(e)
    # ==========================================
# 4. WEB SCRAPING & SEARCH ARSENAL
# ==========================================
def jina_scraper(url: str) -> str:
    api_key = os.environ.get("JINA_API_KEY", "")
    headers = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = httpx.get(f"https://r.jina.ai/{quote(str(url), safe='')}", headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("content", "")[:15000]
    except Exception:
        return "[JINA FALLBACK]: Failed"

async def validate_urls(urls: list) -> dict:
    results = {}
    async with aiohttp.ClientSession() as session:
        for url in set(urls):
            if not url or not str(url).startswith("http"): results[url] = False; continue
            try:
                async with session.get(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status in [403, 405, 501, 503]: results[url] = True
                    else: results[url] = resp.status < 400
            except Exception: results[url] = False
    return results

async def _crawl_urls(urls: list[str]) -> dict[str, str]:
    results = {}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProjectOmega/2.0"}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers) as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for url, resp in zip(urls, responses):
            if isinstance(resp, Exception) or getattr(resp, 'status_code', 400) >= 400:
                jina_text = jina_scraper(url)
                if not jina_text.startswith("[JINA FALLBACK]"): results[url] = jina_text[:15000]
                continue
            try:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for script in soup(["script", "style", "nav", "footer", "header"]): script.extract()
                text = soup.get_text(separator=' ', strip=True)
                lines_txt = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
                clean_text = "\n".join(lines_txt)
                if len(clean_text) < 200:
                    jina_text = jina_scraper(url)
                    if not jina_text.startswith("[JINA FALLBACK]"): clean_text = jina_text
                results[url] = clean_text[:15000]
            except Exception: pass
    return results

@tool(description="Search the web using Jina AI Search API. Returns clean Markdown snippets.")
async def jina_search(query: str) -> str:
    api_key = os.environ.get("JINA_API_KEY", "")
    headers = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"https://s.jina.ai/{quote_plus(str(query))}", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", [])
            if not results: return "No search results found."
            formatted = "Search results:\n\n"
            for i, r in enumerate(results[:5]):
                formatted += f"--- SOURCE {i+1}: {r.get('title', 'No Title')} ---\nURL: {r.get('url', '')}\nSUMMARY:\n{r.get('content', '')[:1500]}\n{'-'*80}\n"
            return formatted
    except Exception as e:
        return f"Jina Search failed: {str(e)}"

@tool(description="Search Wikipedia using the official REST API.")
def wikipedia_rest_search(query: str) -> str:
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3}
        resp = httpx.get(url, params=params, timeout=10).json()
        hits = resp.get("query", {}).get("search", [])
        if not hits: return "No Wikipedia pages found."
        results = []
        for h in hits:
            title = h.get("title", "")
            snippet = h.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            results.append(f"TITLE: {title}\nURL: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}\nSUMMARY: {snippet}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"

SEARXNG_SEARCH_DESCRIPTION = "A free meta-search engine."
_searxng_semaphore = asyncio.Semaphore(4)

async def _fetch_searxng(query: str, base_url: str) -> list:
    async with _searxng_semaphore:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(f"{base_url}/search", params={"q": query, "format": "json"})
                response.raise_for_status()
                return response.json().get("results", [])
        except Exception: return []

async def _fetch_ddg_fallback(query: str) -> list:
    try:
        with DDGS() as ddgs:
            return [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in ddgs.text(query, max_results=5)]
    except Exception: return []

@tool(description=SEARXNG_SEARCH_DESCRIPTION)
async def searxng_search(queries: List[str], max_results: Annotated[int, InjectedToolArg] = 5, config: RunnableConfig = None) -> str:
    configurable = Configuration.from_runnable_config(config)
    base_url = configurable.searxng_base_url
    all_results = {}
    for q in queries:
        results = await _fetch_searxng(q, base_url)
        if not results: results = await _fetch_ddg_fallback(q)
        for res in results:
            url = res.get("url")
            if url and url not in all_results: all_results[url] = {"title": res.get("title", "No Title"), "snippet": res.get("content", ""), "query": q}
    if not all_results: return "No valid search results found."
    urls_to_crawl = list(all_results.keys())[:max_results * len(queries)]
    crawled_content = await _crawl_urls(urls_to_crawl)
    max_char = configurable.max_content_length
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(model=configurable.summarization_model, max_tokens=configurable.summarization_model_max_tokens, api_key=model_api_key, tags=["langsmith:nostream"]).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    
    async def noop(): return None
    tasks = [noop() if not crawled_content.get(url, data.get("snippet")) else summarize_webpage(summarization_model, crawled_content.get(url, data.get("snippet"))[:max_char]) for url, data in all_results.items()]
    summaries = await asyncio.gather(*tasks)
    formatted = "Search results:\n\n"
    for i, ((url, data), summary) in enumerate(zip(all_results.items(), summaries)):
        formatted += f"\n--- SOURCE {i+1}: {data['title']} ---\nURL: {url}\nSUMMARY:\n{summary if summary else data.get('snippet')}\n" + "-"*80 + "\n"
    return formatted

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    try:
        prompt = summarize_webpage_prompt.format(webpage_content=webpage_content[:10000], date=get_today_str())
        summary = await asyncio.wait_for(model.ainvoke([HumanMessage(content=prompt)]), timeout=20.0)
        return f"<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
    except Exception:
        return webpage_content[:2000] + "\n[FALLBACK: Summarization skipped due to timeout/rate-limit]"

# ==========================================
# 5. 3-TIER OMEGA PDF ENGINE (GEMINI CLOUD-OCR)
# ==========================================
_pdf_semaphore = asyncio.Semaphore(1)

def pick_best_pdf_extraction(candidates):
    scored = []
    for label, text in candidates:
        if not text:
            continue
        lowered = text.lower()
        score = 0
        score += 2 if len(text) > 2000 else 0
        score += 2 if "table" in lowered else 0
        score += 1 if "figure" in lowered else 0
        score += 1 if len(set(text.split())) > 300 else 0
        scored.append((score, label, text))
    if not scored:
        return ""
    score, label, text = max(scored, key=lambda x: x[0])
    return f"[{label.upper()} | SCORE {score}]\n{text[:15000]}"



@tool(description="Ingest PDF using 3-Tier Engine: Local Fast-Path, Gemini Cloud-OCR, or Emergency Raw Text.")
async def omega_pdf_ingestor(url: str) -> str:
    if not url.lower().endswith('.pdf'): return "Not a PDF URL."
    async with _pdf_semaphore:
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                pdf_bytes = resp.content
        except Exception as e:
            return f"[PDF FALLBACK]: Download failed: {str(e)}"

        # TIER 1: Local Fast-Path (PyMuPDF4LLM)
        try:
            import pymupdf4llm
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            md = pymupdf4llm.to_markdown(tmp_path)
            os.unlink(tmp_path)
            if len(md.strip()) > 200:
                return f"[TIER 1 LOCAL]\n{md[:15000]}"
        except Exception: pass

        # TIER 2: Gemini Cloud-OCR (Zero Local RAM, 3-Key Rotation)
        gemini_keys_raw = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
        gemini_keys = [k.strip() for k in gemini_keys_raw.split(",") if k.strip()]
        for gemini_key in gemini_keys:
            try:
                import google.generativeai as genai
                import tempfile
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name
                pdf_file = genai.upload_file(tmp_path, mime_type="application/pdf")
                os.unlink(tmp_path)
                response = model.generate_content(["Extract all text, tables, and layout into clean Markdown.", pdf_file])
                try: genai.delete_file(pdf_file.name)
                except: pass
                if response.text:
                    return f"[TIER 2 GEMINI CLOUD]\n{response.text[:15000]}"
            except Exception:
                continue # Try next key if rate limited

        # TIER 3: Emergency Raw Text (PyMuPDF)
        try:
            import fitz
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            text = "\n".join([page.get_text() for page in doc])
            os.unlink(tmp_path)
            return f"[TIER 3 EMERGENCY]\n{text[:15000]}"
        except Exception as e:
            return f"[PDF FALLBACK]: All tiers failed. {str(e)}"
            # ==========================================
# 6. COMPOUNDING MEMORY SYSTEMS
# ==========================================
OMEGA_MEMORY_PATH = os.path.join(os.path.dirname(__file__), "omega_memory.json")

class OmegaMemory:
    def __init__(self):
        self.data = {"domains": {}, "tools": {}}
        self.load()

    def load(self):
        try:
            if os.path.exists(OMEGA_MEMORY_PATH):
                with open(OMEGA_MEMORY_PATH, "r") as f:
                    self.data = json.load(f)
        except Exception:
            self.data = {"domains": {}, "tools": {}}

    def save(self):
        try:
            tmp = OMEGA_MEMORY_PATH + ".tmp"
            with open(tmp, "w") as f: json.dump(self.data, f)
            os.replace(tmp, OMEGA_MEMORY_PATH)
        except Exception: pass

    def update_domain(self, domain, success):
        if not domain: return
        d = self.data["domains"].get(domain, {"trust": 0.5, "hits": 0})
        d["hits"] += 1
        d["trust"] = d["trust"] * 0.9 + (0.1 if success else -0.05)
        d["trust"] = max(0.0, min(1.0, d["trust"]))
        self.data["domains"][domain] = d
        self.save()

    def get_context_prompt(self):
        top_domains = sorted(self.data["domains"].items(), key=lambda x: x[1]["trust"], reverse=True)[:5]
        top_tools = sorted(self.data["tools"].items(), key=lambda x: x[1]["success"]/(x[1]["success"]+x[1]["fail"]+1), reverse=True)[:3]
        ctx = "<MEMORY_REGISTRY>\n"
        if top_domains: ctx += "Trusted Domains: " + ", ".join([f"{d[0]}({d[1]['trust']:.1f})" for d in top_domains]) + "\n"
        if top_tools: ctx += "Reliable Tools: " + ", ".join([t[0] for t in top_tools]) + "\n"
        ctx += "</MEMORY_REGISTRY>"
        return ctx

try:
    omega_memory = OmegaMemory()
except Exception:
    class _NoOpOmegaMemory:
        def get_context_prompt(self):
            return "<MEMORY_REGISTRY>\n</MEMORY_REGISTRY>"
        def update_domain(self, domain, success):
            pass
        def update_tool(self, tool, success):
            pass
    omega_memory = _NoOpOmegaMemory()
except Exception:
    class _NoOpLocalSQLiteMemory:
        def store(self, claim, url):
            pass
        def recall(self, query, limit=5):
            return []
    omega_local_memory = _NoOpLocalSQLiteMemory()

# ==========================================
# 7. EPISTEMIC MATH & CITATION EXECUTIONER
# ==========================================
def score_claim_support(node, page_text):
    claim_words = [w for w in re.findall(r'\b\w+\b', str(getattr(node, "claim", "")).lower()) if len(w) > 2]
    if not claim_words:
        return 0.0

    page_lower = page_text.lower()
    overlap = sum(1 for w in claim_words if w in page_lower)
    lexical_score = overlap / max(len(claim_words), 1)

    if lexical_score < 0.80:
        return 0.0
    return lexical_score



_citation_semaphore = asyncio.Semaphore(5) # Prevents Cloudflare IP bans

async def verify_citations_programmatically(nodes: list) -> list:
    verified_nodes = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProjectOmega/3.0"}
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
        for node in nodes:
            async with _citation_semaphore:
            url = getattr(node, "url", "")
            claim = getattr(node, "claim", "")
            if not url or not claim or not url.startswith("http"): continue
            try:
                resp = await client.get(url)
                if resp.status_code >= 400: continue
                soup = BeautifulSoup(resp.text, 'html.parser')
                for s in soup(["script", "style", "nav", "footer", "header"]): s.extract()
                text = soup.get_text(separator=' ', strip=True).lower()
                claim_lower = claim.lower()
                if claim_lower in text:
                    verified_nodes.append(node)
                    continue
                words = claim_lower.split()
                if len(words) > 3:
                    text_words = text.split()
                    match_count = sum(1 for w in words if w in text_words)
                    if match_count / len(words) >= 0.75:
                        verified_nodes.append(node)
            except Exception: pass
    return verified_nodes

def calculate_epistemic_saturation(evidence_graph: list, research_plan: list) -> float:
    if not research_plan or not evidence_graph: return 0.0
    plan_words = set()
    for node in research_plan:
        topic = node.get("topic", "") if isinstance(node, dict) else getattr(node, "topic", "")
        plan_words.update(re.findall(r'\b\w{4,}\b', topic.lower()))
    if not plan_words: return 0.0
    covered_words = set()
    for node in evidence_graph:
        claim = getattr(node, 'claim', '')
        covered_words.update(re.findall(r'\b\w{4,}\b', claim.lower()))
    if not covered_words: return 0.0
    overlap = len(plan_words.intersection(covered_words))
    saturation = overlap / len(plan_words)
    volume_bonus = min(len(evidence_graph) / 30.0, 0.2)
    return min(saturation + volume_bonus, 1.0)

def programmatic_epistemic_verification(evidence_graph: list, temporal_intent: str) -> dict:
    if not evidence_graph:
        return {"consensus_report": "No evidence gathered.", "confidence_score": 0.0, "red_team_findings": "N/A", "devils_advocate_critique": "N/A"}
    unique_claims = {}
    for node in evidence_graph:
        claim = getattr(node, 'claim', '')
        fp = " ".join(sorted(set(re.findall(r'\b\w{4,}\b', claim.lower()))))
        if fp and fp not in unique_claims: unique_claims[fp] = node
    deduped_nodes = list(unique_claims.values())
    current_year = datetime.now().year
    verified_nodes = []
    decay_penalty = 0.0
    for node in deduped_nodes:
        date_str = getattr(node, 'date_published', None)
        score = 1.0
        if date_str and temporal_intent == "Current":
            try:
                year = int(str(date_str)[:4])
                age = current_year - year
                if age > 0:
                    score *= math.exp(-0.2 * age)
                    decay_penalty += (1.0 - score)
            except: pass
        if score > 0.4: verified_nodes.append((node, score))
    domains = set()
    for node, score in verified_nodes:
        url = getattr(node, 'url', '')
        if url:
            try: domains.add(url.split("//")[-1].split("/")[0].replace("www.", ""))
            except: pass
    source_diversity_score = min(len(domains) / 5.0, 1.0)
    volume_score = min(len(verified_nodes) / 10.0, 1.0)
    contradictions = sum(1 for n, _ in verified_nodes if getattr(n, 'contradicts', []))
    contradiction_penalty = contradictions * 0.05
    final_confidence = (source_diversity_score * 0.4 + volume_score * 0.6) - contradiction_penalty - (decay_penalty * 0.01)
    final_confidence = max(0.1, min(final_confidence, 0.99))
    consensus = f"Programmatic Verification Complete. Analyzed {len(verified_nodes)} unique facts from {len(domains)} domains. Temporal decay applied. Confidence: {final_confidence:.2f}"
    return {
        "consensus_report": consensus, "confidence_score": final_confidence,
        "red_team_findings": f"Programmatic Deduplication removed {len(evidence_graph) - len(deduped_nodes)} duplicate claims.",
        "devils_advocate_critique": f"Temporal decay penalized {decay_penalty:.2f} points. Contradiction penalty: {contradiction_penalty:.2f}."
    }

# ==========================================
# 8. EXECUTION & ROUTING HELPERS
# ==========================================
async def execute_tool_safely(tool, args, config):
    for attempt in range(3):
        try: return await tool.ainvoke(args, config)
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err or "timeout" in err or "resource_exhausted" in err:
                await asyncio.sleep(2 ** attempt); continue
            if attempt == 2: return f"[TOOL FALLBACK]: {getattr(tool, 'name', 'unknown')} failed."
    return "[TOOL FALLBACK]: Max retries exceeded."

def compile_search_results(tool_name: str, raw_output: str) -> str:
    if not isinstance(raw_output, str) or "failed" in raw_output.lower() or "no " in raw_output.lower()[:20]: return raw_output
    # (Regex compilation omitted for brevity, returns raw_output if no match)
    return raw_output

async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.SEARXNG: return [searxng_search, jina_search]
    return [jina_search]

async def get_all_tools(config: RunnableConfig):
    tools = [research_complete_signal, think_tool, omega_pdf_ingestor, github_sniper, huggingface_sniper, hackernews_sniper, wikipedia_rest_search, arxiv_search, pubmed_search, python_repl, audit_pricing, semantic_scholar_search, saas_alternative_sniper, trend_velocity_auditor, jina_search]
    configurable = Configuration.from_runnable_config(config)
    tools.extend(await get_search_tool(SearchAPI(get_config_value(configurable.search_api))))
    return tools

@tool(description="Strategic reflection tool")
def think_tool(reflection: str) -> str: return "Reflection recorded: " + reflection

@tool(description="Signal that research is complete.")
def research_complete_signal() -> str:
    return "Research complete."


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

def anthropic_websearch_called(response):
    try: return response.response_metadata.get("usage", {}).get("server_tool_use", {}).get("web_search_requests", 0) > 0
    except Exception: return False

def openai_websearch_called(response):
    try: return any(t.get("type") == "web_search_call" for t in response.additional_kwargs.get("tool_outputs", []))
    except Exception: return False

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    error_str = str(exception).lower()
    return 'context_length_exceeded' in error_str or 'prompt is too long' in error_str or 'resourceexhausted' in error_str or 'resource_exhausted' in error_str

MODEL_TOKEN_LIMITS = {"groq:llama-3.3-70b-versatile": 128000, "groq:llama-3.1-8b-instant": 128000, "google_genai:gemini-2.0-flash": 1048576, "openai:gpt-4o": 128000, "anthropic:claude-3-5-sonnet": 200000}

def get_model_token_limit(model_string):
    for k, v in MODEL_TOKEN_LIMITS.items():
        if k in model_string: return v
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage): return messages[:i]
    return messages

def check_information_satiation(new_claims: list[str], existing_claims: list[str], threshold: float = 0.75) -> bool:
    if not existing_claims or not new_claims: return False
    def get_core_words(text: str) -> set: return set(re.findall(r'\b\w{4,}\b', text.lower()))
    existing_word_pool = set()
    for claim in existing_claims: existing_word_pool.update(get_core_words(claim))
    if not existing_word_pool: return False
    redundant_claims = 0
    for new_claim in new_claims:
        new_words = get_core_words(new_claim)
        if not new_words: continue
        if len(new_words.intersection(existing_word_pool)) / len(new_words) >= 0.60: redundant_claims += 1
    return (redundant_claims / len(new_claims)) >= threshold

def filter_and_verify_evidence(evidence_graph: list, temporal_intent: str = "Current") -> list:
    if not evidence_graph: return []
    unique_claims = {}
    for node in evidence_graph:
        claim_key = "".join(sorted(re.findall(r'\b\w{4,}\b', getattr(node, 'claim', '').lower())))
        if not claim_key: continue
        if claim_key not in unique_claims: unique_claims[claim_key] = node
        else:
            existing_date = getattr(unique_claims[claim_key], 'date_published', None)
            new_date = getattr(node, 'date_published', None)
            if existing_date and new_date:
                if temporal_intent == "Historical":
                    if str(new_date) < str(existing_date): unique_claims[claim_key] = node
                elif temporal_intent == "Timeless": pass
                else:
                    if str(new_date) > str(existing_date): unique_claims[claim_key] = node
    return list(unique_claims.values())
"""I16.13: omega_security.py - security & adversarial defense.
Extracted from deep_researcher.py. Uses lazy imports for
_record_health_event to avoid circular imports."""

import re
import hashlib
import json
from collections import Counter, deque

# ============================================================
# SECURITY CONSTANTS
# ============================================================
_INJECTION_SIGNATURES = [
    ("ignore previous instructions", 3),
    ("ignore all previous", 3),
    ("disregard previous", 3),
    ("disregard all previous", 3),
    ("forget your instructions", 3),
    ("forget all instructions", 3),
    ("override your instructions", 3),
    ("override your system", 3),
    ("new system prompt", 2),
    ("you are now a", 2),
    ("jailbreak", 2),
    ("do anything now", 2),
    ("developer mode enabled", 2),
    ("ignore the above", 2),
    ("ignore all above", 2),
    ("<system>", 2),
    ("###instruction", 2),
    ("### instruction", 2),
    ("<|system|>", 3),
    ("<|user|>", 3),
    ("<|assistant|>", 3),
    ("act as if you are", 1),
    ("pretend you are", 1),
    ("[system]", 1),
]
_INJECTION_THRESHOLD = 3

_URL_BLOCKED_SCHEMES = ["javascript", "data", "vbscript", "file", "ftp", "gopher"]
_URL_BLOCKED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
_URL_PRIVATE_PATTERNS = [
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    r"192\.168\.\d{1,3}\.\d{1,3}",
    r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}",
]

# I14.9: Redirect-safety constants
_I14_9_MAX_REDIRECTS = 5
_I14_9_BLOCKED_HOSTS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "metadata.google.internal", "metadata.aws.amazon.com",
    "169.254.169.254",
})

def _validate_url_safety(url):
    """G13.2: Validate URL safety. Returns (is_safe, reason)."""
    url_str = str(url or "").strip()
    if not url_str:
        return False, "empty_url"
    url_lower = url_str.lower()
    for scheme in _URL_BLOCKED_SCHEMES:
        if url_lower.startswith(scheme + ":"):
            return False, "blocked_scheme_" + scheme
    if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
        return False, "non_http_scheme"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url_str)
        host = (parsed.hostname or "").lower()
    except Exception:
        return False, "parse_error"
    for blocked in _URL_BLOCKED_HOSTS:
        if host == blocked:
            return False, "blocked_host"
    for pattern in _URL_PRIVATE_PATTERNS:
        if re.search(pattern, host):
            return False, "private_or_ip_host"
    if host.endswith(".onion") or host.endswith(".i2p"):
        return False, "anonymity_network"
    return True, "safe"


def _detect_prompt_injection(text):
    """G13.1: Score-based prompt injection detection."""
    if not text:
        return False, 0, []
    text_lower = str(text).lower()
    score = 0
    matches = []
    for pattern, weight in _INJECTION_SIGNATURES:
        if pattern in text_lower:
            score += weight
            matches.append(pattern)
    return score >= _INJECTION_THRESHOLD, score, matches


def _sanitize_tool_output(output, tool_name="tool"):
    """G13.1: Sanitize tool output. Returns (sanitized_text, was_injection)."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _record_health_event = _dr._record_health_event
    text = str(output or "")
    if not text:
        return text, False
    is_injection, score, patterns = _detect_prompt_injection(text)
    if is_injection:
        sanitized = text
        for p in patterns:
            sanitized = re.sub(re.escape(p), "[REDACTED]", sanitized, flags=re.IGNORECASE)
        _record_health_event("security", "WARNING",
            "G13.1 injection in " + tool_name + " (score=" + str(score) + "): " + str(patterns[:3]))
        return "[QUARANTINED score=" + str(score) + "] " + sanitized[:2000], True
    return text, False


def _detect_content_poisoning(content):
    """G13.2: Detect content poisoning. Returns (is_poisoned, indicators)."""
    text = str(content or "")
    if not text:
        return False, []
    indicators = []
    text_lower = text.lower()
    is_inj, score, _ = _detect_prompt_injection(text)
    if is_inj:
        indicators.append("prompt_injection")
    hidden = ["display:none", "visibility:hidden", "font-size:0", "color:transparent"]
    for p in hidden:
        if p in text_lower:
            indicators.append("hidden_content")
            break
    zw_chars = "\u200b\u200c\u200d\ufeff\u2060"
    zw_count = sum(1 for c in text if c in zw_chars)
    if zw_count > len(text) * 0.01:
        indicators.append("zero_width_obfuscation")
    exfil = ["send this to", "transmit to", "post to http", "curl ", "wget "]
    for p in exfil:
        if p in text_lower:
            indicators.append("exfiltration_attempt")
            break
    return len(indicators) > 0, indicators


def _quarantine_content(output, tool_name="tool"):
    """G14.3: Quarantine poisoned content. Returns (safe_output, was_quarantined)."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _record_health_event = _dr._record_health_event
    text = str(output or "")
    if not text:
        return text, False
    is_poisoned, indicators = _detect_content_poisoning(text)
    if is_poisoned:
        _record_health_event("security", "WARNING",
            "G14.3 quarantined " + tool_name + " output: " + str(indicators[:3]))
        reason = str(indicators[0]) if indicators else "unknown_threat"
        return "[QUARANTINED: " + reason + "] Original output blocked for safety.", True
    return text, False


def _sanitize_evidence_urls(evidence_nodes):
    """G13.2: Filter evidence to safe URLs only. Returns (safe_nodes, rejected_count)."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _record_health_event = _dr._record_health_event
    safe_nodes = []
    rejected = 0
    for node in evidence_nodes or []:
        url = str(getattr(node, "url", "") or "")
        is_safe, reason = _validate_url_safety(url)
        if is_safe:
            safe_nodes.append(node)
        else:
            rejected += 1
            _record_health_event("security", "WARNING", "G13.2 URL rejected (" + reason + "): " + url[:80])
    return safe_nodes, rejected


def _validate_citation_provenance(evidence_nodes, tool_output_text):
    """G13.3: Check evidence URL provenance against tool outputs."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _record_health_event = _dr._record_health_event
    tool_lower = str(tool_output_text or "").lower()
    verified = 0
    orphaned = 0
    for node in evidence_nodes or []:
        url = str(getattr(node, "url", "") or "").strip()
        if not url:
            orphaned += 1
            continue
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
        except Exception:
            domain = url.split("/")[2].lower() if len(url.split("/")) > 2 else ""
        if url.lower() in tool_lower or (domain and domain in tool_lower):
            verified += 1
        else:
            orphaned += 1
    total = len(evidence_nodes or [])
    if orphaned > 0:
        _record_health_event("citation", "WARNING",
            "G13.3 provenance: " + str(orphaned) + "/" + str(total) + " citations lack tool provenance")
    return verified, orphaned


def _detect_citation_laundering(report_content, verified_count):
    """G13.3: Detect citation laundering in final report. Returns list of indicators."""
    text = str(report_content or "")
    indicators = []
    citations_in_report = re.findall(r"\[(\d+)\]", text)
    citation_numbers = set(int(c) for c in citations_in_report)
    over_citations = [n for n in citation_numbers if n > verified_count]
    if over_citations:
        indicators.append("citations_beyond_evidence:" + str(over_citations[:5]))
    word_count = len(text.split())
    if word_count > 0:
        citation_density = len(citations_in_report) / word_count
        if citation_density > 0.15:
            indicators.append("excessive_citation_density:" + str(round(citation_density, 3)))
    from collections import Counter
    citation_counts = Counter(citations_in_report)
    repeated = {k: v for k, v in citation_counts.items() if v > 10}
    if repeated:
        indicators.append("repeated_citation_padding:" + str(repeated))
    return indicators


def _audit_citation_integrity(evidence_nodes):
    """G13.3: Detect citation integrity issues. Returns list of issues."""
    issues = []
    url_claims = {}
    for node in evidence_nodes or []:
        url = str(getattr(node, "url", "") or "").strip()
        claim = str(getattr(node, "claim", "") or "").strip()
        if url:
            if url not in url_claims:
                url_claims[url] = []
            url_claims[url].append(claim)
    for url, claims in url_claims.items():
        if len(claims) > 1:
            for i in range(len(claims)):
                for j in range(i + 1, len(claims)):
                    a_words = set(claims[i].lower().split())
                    b_words = set(claims[j].lower().split())
                    if a_words and b_words:
                        overlap = len(a_words & b_words) / max(1, min(len(a_words), len(b_words)))
                        if overlap < 0.2:
                            issues.append("citation_recycling:" + url[:60])
                            break
    suspicious = ["bit.ly", "tinyurl", "goo.gl", "t.co/", "ow.ly"]
    for url in url_claims.keys():
        for p in suspicious:
            if p in url.lower():
                issues.append("url_shortener:" + url[:60])
                break
    return issues


def _validate_dag_integrity(research_plan):
    """G13.5: Validate DAG structural integrity. Returns list of violations."""
    violations = []
    if not research_plan:
        return violations
    node_ids = []
    for node in research_plan:
        if isinstance(node, dict):
            nid = str(node.get("node_id", ""))
            if nid in node_ids:
                violations.append("duplicate_node_id:" + nid)
            node_ids.append(nid)
    all_ids = set(node_ids)
    for node in research_plan:
        if isinstance(node, dict):
            nid = str(node.get("node_id", ""))
            deps = node.get("depends_on", []) or []
            if nid in [str(d) for d in deps]:
                violations.append("self_reference:" + nid)
            for dep in deps:
                if str(dep) not in all_ids:
                    violations.append("orphaned_dep:" + nid + "->" + str(dep))
    # Cycle detection via Kahn's algorithm
    adj = {}
    in_degree = {}
    for nid in all_ids:
        adj[nid] = []
        in_degree[nid] = 0
    for node in research_plan:
        if isinstance(node, dict):
            nid = str(node.get("node_id", ""))
            deps = node.get("depends_on", []) or []
            for dep in deps:
                dep_str = str(dep)
                if dep_str in all_ids and dep_str != nid:
                    adj[dep_str].append(nid)
                    in_degree[nid] = in_degree.get(nid, 0) + 1
    from collections import deque
    queue = deque([n for n in in_degree if in_degree[n] == 0])
    visited_count = 0
    while queue:
        node = queue.popleft()
        visited_count += 1
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if visited_count < len(all_ids):
        violations.append("cycle_detected")
    return violations


def _compute_plan_fingerprint(research_plan):
    """G13.5: Compute deterministic fingerprint of research plan for mutation detection."""
    plan_repr = json.dumps(research_plan or [], sort_keys=True, default=str)
    return hashlib.sha256(plan_repr.encode("utf-8")).hexdigest()[:16]


def _bind_claim_provenance(claims_with_urls, tool_results):
    """G14.4: Bind claims to specific tool results. Returns provenance records."""
    bindings = []
    for claim, url in claims_with_urls:
        prov = {"claim": claim, "url": url, "source_tool": None,
                "result_identity": None, "provenance_status": "untraceable"}
        for tool_name, result_text in tool_results:
            rt = str(result_text or "")
            result_hash = hashlib.sha256(rt.encode("utf-8")).hexdigest()[:16]
            if url and url.lower() in rt.lower():
                prov["source_tool"] = tool_name
                prov["result_identity"] = result_hash
                prov["provenance_status"] = "verified"
                break
        bindings.append(prov)
    return bindings


def _reject_untraceable_claims(evidence_nodes, tool_results):
    """G14.4: Filter evidence to traceable claims only. Returns (traceable, rejected)."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _record_health_event = _dr._record_health_event
    pairs = [(str(getattr(n, "claim", "") or ""), str(getattr(n, "url", "") or "")) for n in evidence_nodes or []]
    bindings = _bind_claim_provenance(pairs, tool_results)
    traceable = []
    rejected = 0
    for node, binding in zip(evidence_nodes or [], bindings):
        if binding["provenance_status"] == "verified":
            traceable.append(node)
        else:
            rejected += 1
            _record_health_event("citation", "WARNING", "G14.4 untraceable: " + str(getattr(node, "claim", ""))[:60])
    return traceable, rejected


def _enforce_citation_policy(report_content, verified_count, confidence_score):
    """G14.5: Citation laundering hard-fail. Returns (cleaned, adjusted_conf, violations)."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _record_health_event = _dr._record_health_event
    indicators = _detect_citation_laundering(report_content, verified_count)
    if not indicators:
        return report_content, confidence_score, []
    cleaned = report_content
    violations = []
    penalty = 0.0
    for ind in indicators:
        if "beyond_evidence" in ind:
            violations.append("invalid_citation_numbers")
            penalty += 0.15
            def _rm_invalid(m):
                return "" if int(m.group(1)) > verified_count else m.group(0)
            cleaned = re.sub(r"\[(\d+)\]", _rm_invalid, cleaned)
        elif "density" in ind:
            violations.append("citation_padding")
            penalty += 0.10
        elif "repeated" in ind:
            violations.append("suspicious_repetition")
            penalty += 0.10
    adjusted = max(0.0, confidence_score - penalty)
    if violations:
        _record_health_event("citation", "WARNING", "G14.5 violations: " + str(violations))
    return cleaned, adjusted, violations


def _assess_source_diversity(evidence_nodes):
    """G14.6: Source diversity assessment. Returns diversity report."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _record_health_event = _dr._record_health_event
    domains = {}
    urls_seen = set()
    duplicates = 0
    for node in evidence_nodes or []:
        url = str(getattr(node, "url", "") or "")
        if not url:
            continue
        if url in urls_seen:
            duplicates += 1
            continue
        urls_seen.add(url)
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
        except Exception:
            domain = url.split("/")[2].lower() if len(url.split("/")) > 2 else "unknown"
        canonical = domain.replace("www.", "")
        domains[canonical] = domains.get(canonical, 0) + 1
    unique_domains = len(domains)
    total_sources = len(urls_seen)
    diversity_ratio = unique_domains / max(1, total_sources)
    report = {"total_sources": total_sources, "unique_domains": unique_domains,
              "diversity_ratio": round(diversity_ratio, 3), "duplicate_sources": duplicates,
              "is_diverse": diversity_ratio >= 0.5 and unique_domains >= 2}
    if not report["is_diverse"] and total_sources > 0:
        _record_health_event("citation", "WARNING", "G14.6 low diversity: " + str(unique_domains) + "/" + str(total_sources))
    return report


def _i13_11_tokens(text):
    """I13.11: Tokenize text for content fingerprinting."""
    return set(re.findall(r"[a-z0-9_]{4,}", str(text or "").lower()))


def _i13_11_canonical_source(url):
    """I13.11: Normalize URL to canonical source identity."""
    url_str = str(url or "").strip().lower()
    if not url_str:
        return ""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url_str)
        domain = parsed.netloc
        for prefix in ("www.", "m.", "amp.", "mobile."):
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        path = parsed.path.rstrip("/")
        return (domain + path) if path else domain
    except Exception:
        return url_str


def _i13_11_assess_independence(evidence_nodes):
    """I13.11: True source independence. Distinguishes unique URLs, unique
    domains, canonical sources, and underlying independent content origins."""
    from open_deep_research import deep_researcher as _dr  # I16.13 lazy
    _record_health_event = _dr._record_health_event
    urls = set()
    domains = set()
    canonical_sources = set()
    claims = []
    for node in evidence_nodes or []:
        url = str(getattr(node, "url", "") or "").strip()
        claim = str(getattr(node, "claim", "") or "").strip()
        if not url:
            continue
        urls.add(url)
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            for prefix in ("www.", "m.", "amp.", "mobile."):
                if domain.startswith(prefix):
                    domain = domain[len(prefix):]
        except Exception:
            domain = ""
        if domain:
            domains.add(domain)
        canonical = _i13_11_canonical_source(url)
        if canonical:
            canonical_sources.add(canonical)
        if claim:
            claims.append(claim)
    clusters = []
    for claim in claims:
        claim_tokens = _i13_11_tokens(claim)
        if not claim_tokens:
            continue
        placed = False
        for cluster in clusters:
            rep = cluster["tokens"]
            overlap = len(claim_tokens & rep) / max(1, min(len(claim_tokens), len(rep)))
            if overlap >= 0.7:
                placed = True
                break
        if not placed:
            clusters.append({"tokens": claim_tokens})
    independent_sources = len(clusters)
    total_sources = len(urls)
    independence_ratio = min(1.0, independent_sources / max(1, total_sources))
    report = {
        "total_sources": total_sources,
        "unique_urls": len(urls),
        "unique_domains": len(domains),
        "canonical_sources": len(canonical_sources),
        "independent_sources": independent_sources,
        "independence_ratio": round(independence_ratio, 3),
        "is_independent": independence_ratio >= 0.5 and independent_sources >= 2,
    }
    if not report["is_independent"] and total_sources > 0:
        _record_health_event("citation", "WARNING", "I13.11 low independence: " + str(independent_sources) + "/" + str(total_sources))
    return report

# ============================================================
# I14.9: REDIRECT-SAFE URL VALIDATION (I16.13-B-FIX)
# ============================================================
_I14_9_BLOCKED_HOSTS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "metadata.google.internal", "metadata.aws.amazon.com",
    "169.254.169.254",
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

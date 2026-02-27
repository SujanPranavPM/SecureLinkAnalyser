"""
feature_extractor.py
====================
Modular, reusable URL feature engineering for malicious URL detection.
Each feature is isolated in its own function for testability and clarity.

Author: Cybersecurity ML Pipeline
"""

import re
import math
from urllib.parse import urlparse
from typing import Dict, List, Any
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "paypal", "password", "credential", "ebay", "amazon", "apple",
    "microsoft", "google", "facebook", "instagram", "netflix", "free",
    "winner", "prize", "click", "urgent", "suspend", "limited", "offer",
    "cheap", "pharmacy", "casino", "porn", "adult", "hack", "crack",
    "keygen", "warez", "phish", "malware",
]

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

SHORTENING_SERVICES = [
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "rebrand.ly",
]

TRUSTED_TLDS = {".com", ".org", ".edu", ".gov", ".net", ".io", ".co"}


# ---------------------------------------------------------------------------
# Individual Feature Extractors
# ---------------------------------------------------------------------------

def get_url_length(url: str) -> int:
    """Total character length of the URL."""
    return len(url)


def get_domain_length(parsed) -> int:
    """Character length of the domain/netloc."""
    return len(parsed.netloc)


def get_path_length(parsed) -> int:
    """Character length of the URL path."""
    return len(parsed.path)


def get_num_dots(url: str) -> int:
    """Count of dots in the full URL."""
    return url.count(".")


def get_num_hyphens(url: str) -> int:
    """Count of hyphen characters."""
    return url.count("-")


def get_num_underscores(url: str) -> int:
    """Count of underscores."""
    return url.count("_")


def get_num_slashes(url: str) -> int:
    """Count of forward slashes (excluding protocol)."""
    return url.count("/")


def get_num_at_signs(url: str) -> int:
    """Presence of @ symbol (common in phishing redirects)."""
    return url.count("@")


def get_num_question_marks(url: str) -> int:
    """Count of query parameter separators."""
    return url.count("?")


def get_num_equals(url: str) -> int:
    """Count of = signs in query string."""
    return url.count("=")


def get_num_ampersands(url: str) -> int:
    """Count of & signs in query string."""
    return url.count("&")


def get_num_percent(url: str) -> int:
    """Count of % (URL encoding, often used to obfuscate)."""
    return url.count("%")


def get_num_special_chars(url: str) -> int:
    """Count of all special characters."""
    return len(re.findall(r"[^a-zA-Z0-9\.\-\_\/\:\?=&#]", url))


def get_num_digits(url: str) -> int:
    """Count of digit characters in URL."""
    return sum(c.isdigit() for c in url)


def has_ip_address(parsed) -> int:
    """1 if domain appears to be an IP address, 0 otherwise."""
    host = parsed.hostname or ""
    return int(bool(IP_PATTERN.match(host)))


def has_https(parsed) -> int:
    """1 if URL uses HTTPS, 0 otherwise."""
    return int(parsed.scheme == "https")


def get_num_subdomains(parsed) -> int:
    """Count of subdomains (dots in hostname minus 1)."""
    host = parsed.hostname or ""
    parts = host.split(".")
    return max(0, len(parts) - 2)


def has_suspicious_keywords(url: str) -> int:
    """1 if URL contains any known phishing/suspicious keyword."""
    lower = url.lower()
    return int(any(kw in lower for kw in SUSPICIOUS_KEYWORDS))


def count_suspicious_keywords(url: str) -> int:
    """Total count of suspicious keyword occurrences."""
    lower = url.lower()
    return sum(lower.count(kw) for kw in SUSPICIOUS_KEYWORDS)


def is_url_shortener(parsed) -> int:
    """1 if domain belongs to a known URL shortening service."""
    host = parsed.hostname or ""
    return int(any(s in host for s in SHORTENING_SERVICES))


def get_entropy(url: str) -> float:
    """
    Shannon entropy of the URL string.
    High entropy suggests obfuscated or randomized domains.
    """
    if not url:
        return 0.0
    freq = {}
    for c in url:
        freq[c] = freq.get(c, 0) + 1
    probs = [v / len(url) for v in freq.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def get_digit_ratio(url: str) -> float:
    """Ratio of digit characters to total URL length."""
    if not url:
        return 0.0
    return sum(c.isdigit() for c in url) / len(url)


def get_letter_ratio(url: str) -> float:
    """Ratio of letter characters to total URL length."""
    if not url:
        return 0.0
    return sum(c.isalpha() for c in url) / len(url)


def has_double_slash_redirect(url: str) -> int:
    """1 if URL contains // after the protocol (redirect trick)."""
    return int("//" in url[8:])


def get_tld_suspicious(parsed) -> int:
    """1 if TLD is uncommon/potentially suspicious."""
    host = parsed.hostname or ""
    parts = host.split(".")
    tld = "." + parts[-1] if parts else ""
    return int(tld.lower() not in TRUSTED_TLDS)


def get_domain_has_digits(parsed) -> int:
    """1 if the domain itself contains digits (common in generated domains)."""
    host = parsed.hostname or ""
    parts = host.split(".")
    domain = parts[-2] if len(parts) >= 2 else ""
    return int(any(c.isdigit() for c in domain))


# ---------------------------------------------------------------------------
# Master Feature Extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at_signs",
    "num_question_marks",
    "num_equals",
    "num_ampersands",
    "num_percent",
    "num_special_chars",
    "num_digits",
    "has_ip_address",
    "has_https",
    "num_subdomains",
    "has_suspicious_keywords",
    "count_suspicious_keywords",
    "is_url_shortener",
    "entropy",
    "digit_ratio",
    "letter_ratio",
    "has_double_slash_redirect",
    "tld_suspicious",
    "domain_has_digits",
]


def extract_features(url: str) -> Dict[str, Any]:
    """
    Extract all features from a URL.

    Parameters
    ----------
    url : str  — The raw URL string.

    Returns
    -------
    dict — Feature name → value mapping.
    """
    # Ensure URL has a scheme for proper parsing
    if not url.startswith(("http://", "https://")):
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    try:
        parsed = urlparse(url_for_parse)
    except Exception:
        parsed = urlparse("http://invalid.com")

    return {
        "url_length": get_url_length(url),
        "domain_length": get_domain_length(parsed),
        "path_length": get_path_length(parsed),
        "num_dots": get_num_dots(url),
        "num_hyphens": get_num_hyphens(url),
        "num_underscores": get_num_underscores(url),
        "num_slashes": get_num_slashes(url),
        "num_at_signs": get_num_at_signs(url),
        "num_question_marks": get_num_question_marks(url),
        "num_equals": get_num_equals(url),
        "num_ampersands": get_num_ampersands(url),
        "num_percent": get_num_percent(url),
        "num_special_chars": get_num_special_chars(url),
        "num_digits": get_num_digits(url),
        "has_ip_address": has_ip_address(parsed),
        "has_https": has_https(parsed),
        "num_subdomains": get_num_subdomains(parsed),
        "has_suspicious_keywords": has_suspicious_keywords(url),
        "count_suspicious_keywords": count_suspicious_keywords(url),
        "is_url_shortener": is_url_shortener(parsed),
        "entropy": round(get_entropy(url), 4),
        "digit_ratio": round(get_digit_ratio(url), 4),
        "letter_ratio": round(get_letter_ratio(url), 4),
        "has_double_slash_redirect": has_double_slash_redirect(url),
        "tld_suspicious": get_tld_suspicious(parsed),
        "domain_has_digits": get_domain_has_digits(parsed),
    }


def extract_feature_vector(url: str) -> np.ndarray:
    """Return features as a numpy array (ordered by FEATURE_NAMES)."""
    feats = extract_features(url)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


def get_triggered_reasons(features: Dict[str, Any]) -> List[str]:
    """
    Human-readable explanation of which features indicate risk.

    Parameters
    ----------
    features : dict — Output of extract_features()

    Returns
    -------
    list[str] — Reasons list for the API response.
    """
    reasons = []

    if features["url_length"] > 75:
        reasons.append(f"Unusually long URL ({features['url_length']} chars)")
    if features["has_ip_address"]:
        reasons.append("Domain is a raw IP address (not a hostname)")
    if features["num_at_signs"] > 0:
        reasons.append("Contains '@' symbol — possible redirect trick")
    if not features["has_https"]:
        reasons.append("No HTTPS — connection is unencrypted")
    if features["num_subdomains"] >= 3:
        reasons.append(f"Excessive subdomains ({features['num_subdomains']}) — mimicking trusted site")
    if features["has_suspicious_keywords"]:
        reasons.append("Contains phishing-associated keywords (e.g. login, verify, secure)")
    if features["is_url_shortener"]:
        reasons.append("URL shortener used — destination is hidden")
    if features["entropy"] > 4.5:
        reasons.append(f"High URL entropy ({features['entropy']}) — suggests obfuscation")
    if features["num_hyphens"] > 4:
        reasons.append(f"Many hyphens ({features['num_hyphens']}) — typosquatting indicator")
    if features["tld_suspicious"]:
        reasons.append("Uncommon TLD — not a standard trusted extension")
    if features["num_dots"] > 5:
        reasons.append(f"Many dots ({features['num_dots']}) — complex subdomain structure")
    if features["digit_ratio"] > 0.3:
        reasons.append("High proportion of digits — potential generated domain")
    if features["has_double_slash_redirect"]:
        reasons.append("Double-slash redirect pattern detected")
    if features["num_percent"] > 5:
        reasons.append(f"Heavy URL encoding ({features['num_percent']} '%' chars) — obfuscation attempt")

    return reasons if reasons else ["No specific high-risk indicators triggered"]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login/verify?user=admin@bank.com",
        "http://paypal-secure-login.verify-account.com/update/credentials",
        "https://bit.ly/3xYzABC",
    ]
    for u in test_urls:
        feats = extract_features(u)
        reasons = get_triggered_reasons(feats)
        print(f"\nURL: {u}")
        print(f"  Entropy : {feats['entropy']}")
        print(f"  Reasons : {reasons}")

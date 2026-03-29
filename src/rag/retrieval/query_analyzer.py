from __future__ import annotations

import re
from typing import Dict, List, Set


TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ỹ0-9_+-]+", re.UNICODE)

STOPWORDS = {
    # English
    "how", "what", "when", "where", "which", "with", "from", "into", "your",
    "have", "does", "this", "that", "using", "used", "about", "there", "need",
    "then", "than", "they", "them", "onto", "into", "help", "please",
    # Vietnamese
    "làm", "thế", "nào", "để", "bằng", "cách", "mình", "tôi", "có", "thể",
    "và", "của", "cho", "với", "trên", "khi", "sao", "vào", "trong", "như",
    "được", "một", "các", "những", "là", "muốn", "giúp", "lên",
}

GENERIC_TERMS = {
    "project", "projects", "repo", "repository",
    "repositories", "account", "accounts", "user", "users",
}

PLATFORM_TERMS = {"github", "gitlab"}

SPECIAL_VARIANTS = {
    "gitbash": ["gitbash", "git bash", "bash"],
    "passkey": ["passkey", "passkeys"],
    "passkeys": ["passkey", "passkeys"],
    "login": ["login", "log in", "sign in", "signin", "đăng nhập"],
    "signin": ["sign in", "signin", "login", "log in", "đăng nhập"],
    "authentication": ["authentication", "authenticate", "xác thực", "2fa", "two-factor", "two factor"],
    "manage": ["manage", "managing", "management", "quản lý"],
    "push": ["push"],
    "fork": ["fork"],
    "ssh": ["ssh"],
    "handbook": ["handbook", "cẩm nang", "sổ tay"],
    "policy": ["policy", "policies", "chính sách"],
    "support": ["support", "hỗ trợ"],
    "engineering": ["engineering", "kỹ thuật"],
    "access": ["access", "permission", "permissions", "quyền", "truy cập"],
}

AUTH_PHRASES = {
    "passkey",
    "passkeys",
    "login",
    "log in",
    "sign in",
    "signin",
    "đăng nhập",
    "authentication",
    "authenticate",
    "xác thực",
    "2fa",
    "two-factor",
    "two factor",
    "manage passkeys",
    "managing passkeys",
    "manage your passkeys",
    "managing your passkeys",
    "sign in with a passkey",
    "signing in with a passkey",
    "đăng nhập bằng passkey",
    "quản lý passkey",
    "quản lý passkeys",
}

HANDBOOK_PHRASES = {
    "handbook",
    "cẩm nang",
    "sổ tay",
    "support handbook",
    "engineering handbook",
    "handbook policy",
    "policy",
    "policies",
}

REPO_ACCESS_PHRASES = {
    "repository access",
    "repo access",
    "repository permission",
    "repository permissions",
    "quyền truy cập repository",
    "quyền truy cập repo",
    "permission",
    "permissions",
    "access",
    "collaborator",
    "collaborators",
    "team access",
}


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def unique_keep_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase in text


def _contains_any_phrase(text: str, phrases: Set[str]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _expand_term_variants(term: str) -> List[str]:
    term = normalize_text(term)
    variants = {term}

    if term.endswith("e") and len(term) > 3:
        variants.add(term[:-1] + "ing")

    if term.endswith("ing") and len(term) > 5:
        variants.add(term[:-3])
        if not term[:-3].endswith("e"):
            variants.add(term[:-3] + "e")

    if not term.endswith("s") and len(term) > 3:
        variants.add(term + "s")

    if term.endswith("s") and len(term) > 4:
        variants.add(term[:-1])

    return [v for v in variants if v]


def _detect_intents(normalized_query: str, all_keywords: List[str]) -> Dict[str, List[str]]:
    intent_labels: List[str] = []
    preferred_sources: List[str] = []
    penalized_sources: List[str] = []
    phrase_keywords: List[str] = []

    if _contains_any_phrase(normalized_query, AUTH_PHRASES) or any(
        kw in {"passkey", "passkeys", "authentication", "login", "signin"} for kw in all_keywords
    ):
        intent_labels.append("github_authentication")
        preferred_sources.extend(["github_docs"])
        penalized_sources.extend(["gitlab_handbook"])
        phrase_keywords.extend([
            "passkey",
            "passkeys",
            "sign in",
            "sign in with a passkey",
            "signing in with a passkey",
            "manage your passkeys",
            "managing your passkeys",
            "authentication",
            "login",
            "đăng nhập",
            "quản lý passkey",
        ])

    if _contains_any_phrase(normalized_query, HANDBOOK_PHRASES):
        intent_labels.append("gitlab_handbook")
        preferred_sources.extend(["gitlab_handbook"])
        phrase_keywords.extend([
            "handbook",
            "support handbook",
            "engineering handbook",
            "policy",
        ])

    if _contains_any_phrase(normalized_query, REPO_ACCESS_PHRASES):
        intent_labels.append("repository_access")
        preferred_sources.extend(["github_docs", "github_issues"])
        penalized_sources.extend(["gitlab_handbook"])
        phrase_keywords.extend([
            "repository access",
            "permission",
            "permissions",
            "managing teams and people with access to your repository",
            "permission levels for a personal account repository",
        ])

    return {
        "intent_labels": unique_keep_order(intent_labels),
        "preferred_sources": unique_keep_order(preferred_sources),
        "penalized_sources": unique_keep_order(penalized_sources),
        "phrase_keywords": unique_keep_order(phrase_keywords),
    }


def extract_keyword_groups(query: str) -> Dict[str, List[str]]:
    """
    Tách query thành:
    - all_keywords
    - platform_keywords
    - strong_keywords
    - weak_keywords
    - intent_labels
    - preferred_sources
    - penalized_sources
    - phrase_keywords
    """
    raw_tokens = tokenize(query)
    all_keywords: List[str] = []

    for token in raw_tokens:
        if token in STOPWORDS:
            continue

        if len(token) >= 4 or token in {"git", "ssh", "2fa"}:
            all_keywords.append(token)

    normalized_query = normalize_text(query)
    compact_query = normalized_query.replace(" ", "")

    for key, variants in SPECIAL_VARIANTS.items():
        if key in compact_query or key in normalized_query:
            all_keywords.extend(variants)

    # phrase-driven keyword enrichment
    if _contains_any_phrase(normalized_query, AUTH_PHRASES):
        all_keywords.extend([
            "passkey",
            "passkeys",
            "authentication",
            "login",
            "signin",
            "manage",
            "managing",
        ])

    if _contains_any_phrase(normalized_query, HANDBOOK_PHRASES):
        all_keywords.extend(["handbook", "policy", "support", "engineering"])

    if _contains_any_phrase(normalized_query, REPO_ACCESS_PHRASES):
        all_keywords.extend(["repository", "access", "permission", "permissions"])

    expanded_keywords: List[str] = []
    for kw in all_keywords:
        expanded_keywords.extend(_expand_term_variants(kw))

    all_keywords = unique_keep_order(expanded_keywords)

    platform_keywords = [kw for kw in all_keywords if kw in PLATFORM_TERMS]
    non_platform_keywords = [kw for kw in all_keywords if kw not in PLATFORM_TERMS]

    strong_keywords = [kw for kw in non_platform_keywords if kw not in GENERIC_TERMS]
    weak_keywords = [kw for kw in non_platform_keywords if kw in GENERIC_TERMS]

    if not strong_keywords:
        strong_keywords = non_platform_keywords[:]

    intent_info = _detect_intents(normalized_query, all_keywords)

    return {
        "all_keywords": all_keywords,
        "platform_keywords": unique_keep_order(platform_keywords),
        "strong_keywords": unique_keep_order(strong_keywords),
        "weak_keywords": unique_keep_order(weak_keywords),
        "intent_labels": intent_info["intent_labels"],
        "preferred_sources": intent_info["preferred_sources"],
        "penalized_sources": intent_info["penalized_sources"],
        "phrase_keywords": intent_info["phrase_keywords"],
    }
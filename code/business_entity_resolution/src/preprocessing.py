"""
Text and Field Preprocessing for Business Entity Resolution.
Handles legal suffix normalization, address standardization, open-world country handling,
and entity tokenization without external APIs or bundled gazetteers.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple

# Standard business legal suffixes
LEGAL_SUFFIXES = {
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt ltd": "private limited",
    "private limited": "private limited",
    "pvt": "private",
    "private": "private",
    "llc": "llc",
    "l.l.c.": "llc",
    "co": "company",
    "company": "company",
    "sa": "societe anonyme",
    "s.a.": "societe anonyme",
    "sarl": "societe a responsabilite limitee",
    "s.a.r.l.": "societe a responsabilite limitee",
    "gmbh": "gmbh",
    "plc": "public limited company",
}

# Standard address token replacements
ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "dr": "drive",
    "pkwy": "parkway",
    "hwy": "highway",
    "fl": "floor",
    "apt": "apartment",
    "ste": "suite",
    "sq": "square",
    "pl": "place",
    "ln": "lane",
    "ct": "court",
    "ctr": "center",
    "opp": "opposite",
    "nr": "near",
}

# Country mapping for common variations; preserves unknown countries as open set
COUNTRY_ALIASES = {
    "us": "us",
    "usa": "us",
    "u.s.": "us",
    "u.s.a.": "us",
    "united states": "us",
    "united states of america": "us",
    "india": "india",
    "in": "india",
    "ind": "india",
    "bharat": "india",
    "france": "france",
    "fr": "france",
    "fra": "france",
    "republique francaise": "france",
}


def strip_accents(text: str) -> str:
    """Normalize unicode characters (e.g., é -> e, ñ -> n)."""
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def normalize_country(country_str: str) -> str:
    """
    Normalizes country strings while respecting open-world assumption.
    Does NOT restrict only to US/India/France.
    """
    if not country_str or not isinstance(country_str, str):
        return "unknown"
    cleaned = strip_accents(country_str).lower().strip()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return COUNTRY_ALIASES.get(cleaned, cleaned)


def extract_legal_suffix(name: str) -> Tuple[str, str]:
    """
    Extracts and standardizes legal suffix from business name.
    Returns (cleaned_name_without_suffix, detected_canonical_suffix).
    """
    norm_name = f" {name.lower()} "
    for alias, canonical in sorted(
        LEGAL_SUFFIXES.items(), key=lambda x: len(x[0]), reverse=True
    ):
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, norm_name):
            cleaned = re.sub(pattern, "", norm_name)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned, canonical
    return name.strip(), ""


def normalize_business_name(name: str) -> str:
    """
    Cleans and standardizes business name.
    - Lowers case, strips accents
    - Standardizes '&' -> 'and'
    - Standardizes legal abbreviations
    - Strips non-alphanumeric (except single spaces)
    """
    if not name or not isinstance(name, str):
        return ""
    cleaned = strip_accents(name).lower()
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    tokens = cleaned.split()
    processed_tokens = []
    i = 0
    while i < len(tokens):
        # Check two-word suffixes like 'pvt ltd'
        if i + 1 < len(tokens) and f"{tokens[i]} {tokens[i+1]}" in LEGAL_SUFFIXES:
            processed_tokens.append(LEGAL_SUFFIXES[f"{tokens[i]} {tokens[i+1]}"])
            i += 2
        elif tokens[i] in LEGAL_SUFFIXES:
            processed_tokens.append(LEGAL_SUFFIXES[tokens[i]])
            i += 1
        else:
            processed_tokens.append(tokens[i])
            i += 1

    return " ".join(processed_tokens).strip()


def normalize_address(address: str) -> str:
    """
    Cleans and standardizes business address.
    - Expands road/street abbreviations
    - Normalizes separators
    """
    if not address or not isinstance(address, str):
        return ""
    cleaned = strip_accents(address).lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    tokens = cleaned.split()
    standardized = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(standardized).strip()


def extract_address_digits(address: str) -> Set[str]:
    """
    Extracts numeric tokens from address (street numbers, PIN/zip codes).
    """
    if not address or not isinstance(address, str):
        return set()
    return set(re.findall(r"\b\d+\b", address))


def get_character_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from string."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}

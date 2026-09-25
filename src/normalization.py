# src/normalization.py
import re
import unicodedata

# ---------- Legal suffixes to strip from names ----------
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "ltd", "limited", "lp", "llp", "plc",
    "pvt", "private", "opc", "huf", "prop", "proprietor",
}

# ---------- Address abbreviations ----------
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "sq": "square", "hwy": "highway", "pkwy": "parkway",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "apt": "apartment", "ste": "suite", "fl": "floor",
    "dist": "district", "distt": "district",
}

# Words that are just landmark noise in Indian addresses
STOP_ADDR = {"near", "opp", "opposite", "behind", "beside", "adjacent"}

# Country aliases
COUNTRY_MAP = {
    "us": "us", "usa": "us", "united states": "us",
    "united states of america": "us",
    "india": "india", "in": "india", "bharat": "india",
    "france": "france", "fr": "france",
}

# Pincode patterns per country
PIN_PATTERNS = {
    "india": re.compile(r"\b(\d{6})\b"),
    "us": re.compile(r"\b(\d{5})(?:-\d{4})?\b"),
    "france": re.compile(r"\b(\d{5})\b"),
}


def _is_nan(x):
    return x is None or (isinstance(x, float) and x != x) or str(x).lower() == "nan"


def normalize_text(text):
    """Basic cleanup: lowercase, strip punctuation (keep letters/digits/spaces)."""
    if _is_nan(text):
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = s.replace("&", " and ")
    s = "".join(
        c if (c.isalnum() or c.isspace() or unicodedata.category(c).startswith("M"))
        else " "
        for c in s
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_business_name(name):
    """Lowercase, strip punctuation, remove legal suffixes."""
    s = normalize_text(name)
    toks = [t for t in s.split() if t not in LEGAL_SUFFIXES]
    return " ".join(toks)


def normalize_address(address):
    """Expand abbreviations, remove landmark noise."""
    s = normalize_text(address)
    toks = [t for t in s.split() if t not in STOP_ADDR]
    toks = [ADDR_ABBREV.get(t, t) for t in toks]
    return " ".join(toks)


def normalize_country(country):
    s = normalize_text(country)
    return COUNTRY_MAP.get(s, s)


def extract_pincode(address, country):
    """Extract postal code using country-specific pattern, with fallback."""
    if _is_nan(address):
        return ""
    s = unicodedata.normalize("NFKC", str(address))
    c = normalize_country(country)
    if c in PIN_PATTERNS:
        m = PIN_PATTERNS[c].search(s)
        if m:
            return m.group(1)
    # fallback: try all patterns
    for cc in ("india", "france", "us"):
        m = PIN_PATTERNS[cc].search(s)
        if m:
            return m.group(1)
    return ""


def normalize_dataframe(df):
    """Add normalized columns. Do not drop originals."""
    df = df.copy()
    df["business_name_normalized"] = df["business_name"].apply(normalize_business_name)
    df["business_address_normalized"] = df["business_address"].apply(normalize_address)
    df["country_normalized"] = df["country"].apply(normalize_country)
    df["pincode"] = [
        extract_pincode(a, c)
        for a, c in zip(df["business_address"], df["country"])
    ]
    return df


if __name__ == "__main__":
    examples = [
        ("Orelee's Barbershop", ""),
        ("B+ Retail Inc", ""),
        ("Prabhav Business Center", "India"),
        ("797, Lake Town Block A, Kolkata 700089", "India"),
        ("123 Main St, Boston MA 02134", "USA"),
    ]
    for name, country in examples:
        print(f"name:    {name!r} -> {normalize_business_name(name)!r}")
        if country:
            addr = name
            print(f"addr:    {addr!r} -> {normalize_address(addr)!r}  pin={extract_pincode(addr, country)!r}")
        print()
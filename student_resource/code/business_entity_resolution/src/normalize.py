import re
import unicodedata

from unidecode import unidecode


LEGAL_SUFFIXES = {
    "llc", "llp", "inc", "incorporated", "corp", "corporation", "company",
    "co", "limited", "ltd", "private", "pvt", "plc", "pc", "l l c",
}

ABBREVIATIONS = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "route": "rte",
    "apartment": "apt", "suite": "ste", "floor": "fl",
    "madhya pradesh": "mp", "uttar pradesh": "up",
}


def _unicode_text(value: object) -> str:
    if value is None:                                                                                                                                                                                                                                                                                                                                                                                            
        return ""
    value = str(value).strip().lower()
    if value in {"nan", "none", "null"}:
        return ""
    return unicodedata.normalize("NFKC", value)


def normalize_text(value: object) -> str:
    """Language-preserving normalization used for multilingual retrieval."""
    value = _unicode_text(value)
    value = value.replace("&", " and ")
    value = re.sub(r"https?://|www\.", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_latin(value: object) -> str:
    """ASCII transliteration view; original-script normalization is retained separately."""
    return normalize_text(unidecode(_unicode_text(value)))


def normalize_name(value: object, strip_legal: bool = False) -> str:
    value = normalize_latin(value)
    tokens = value.split()
    if strip_legal:
        tokens = [token for token in tokens if token not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_address(value: object) -> str:
    value = normalize_latin(value)
    for source, target in ABBREVIATIONS.items():
        value = re.sub(rf"\b{re.escape(source)}\b", target, value)
    return re.sub(r"\s+", " ", value).strip()


def digits(value: object) -> set[str]:
    return set(re.findall(r"\d+", normalize_latin(value)))


# src/cleaner.py
# OCR text cleaning utilities for Uzbek Islamic texts
import re

# Common Uzbek suffixes that are frequently split off by OCR errors.
# These rarely appear as standalone words.
UZBEK_SUFFIXES = [
    "ning", "dan", "ga", "ka", "qa", "da", "ta", "ni",
    "lar", "ler", "dir", "dur", "dagi", "daki",
    "ida", "iga", "ini", "idan", "ining",
    "imiz", "ingiz", "lari", "larini", "lariga", "larida",
    "si", "ning", "miz", "ngiz",
]

# Known specific OCR errors and their corrections
SPECIFIC_FIXES = [
    (r'\bha\s+qida\b',                  'haqida'),
    (r'\bhaqi\s+da\b',                  'haqida'),
    (r'\bpayg\'ambarim\s+izning\b',     "payg'ambarimizning"),
    (r'\bkutubxon\s+asi\b',             'kutubxonasi'),
    (r'\bkutubxo\s+nasi\b',             'kutubxonasi'),
    (r'\bXuzari\s+y\b',                 'Xuzariy'),
    (r'\bbilim\s+yurtlarida\b',         'bilim yurtlarida'),
    (r'\bqo\'lla\s+niladi\b',           "qo'llaniladi"),
    (r'\bo\'td\s+i\b',                  "o'tdi"),
    (r'\basrnin\s+g\b',                 'asrning'),
    (r'\bbat\s+afsil\b',               'batafsil'),
    (r'\bm\'\s+lumot\b',               "ma'lumot"),
    (r'\bberuvc\s+hi\b',               'beruvchi'),
    (r'\bo\'rnida\b',                   "o'rnida"),
]


def fix_split_words(text):
    """Merge short suffix fragments back onto the preceding word.

    Example: "payg'ambarim izning" -> "payg'ambarimizning"
    Only merges when word2 is 1-2 characters and is not a known independent word.
    """
    def maybe_merge(match):
        word1, space, word2 = match.group(1), match.group(2), match.group(3)
        # Skip merging if word2 is a known independent word
        if word2.lower() in {"va", "bu", "u", "men", "sen", "biz", "siz", "ular",
                              "ham", "shu", "har", "o'z", "bir", "yoki", "agar",
                              "emas", "bor", "yo'q", "edi"}:
            return match.group(0)
        if len(word2) <= 2 and len(word1) >= 4:
            return word1 + word2
        return match.group(0)

    text = re.sub(r'\b([a-zA-Zo\'''\u02bc]{4,})( )([a-z]{1,2})\b',
                  maybe_merge, text)
    return text


def clean_text(text: str) -> str:
    """Apply a multi-step cleaning pipeline to OCR-extracted Uzbek text."""
    # 1. Remove noise lines (URLs, library watermarks, lone page numbers)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'kutubxonasi\s*\d*', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 2. Strip invisible / control characters
    text = text.replace('\u00ad', '')   # soft hyphen
    text = text.replace('\u200b', '')   # zero-width space
    text = text.replace('\ufeff', '')   # BOM

    # 3. Rejoin hyphenated line breaks: "short-\n en" -> "shorten"
    text = re.sub(r'-\s*\n\s*', '', text)

    # 4. Apply known OCR error corrections
    for pattern, replacement in SPECIFIC_FIXES:
        text = re.sub(pattern, replacement, text)

    # 5. Collapse multiple spaces and tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # 6. Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. Merge split suffix fragments (applied conservatively)
    text = fix_split_words(text)

    # 8. Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line)

    return text.strip()

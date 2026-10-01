"""Domain normalization and balanced dataset assembly.

Training labels come from downloaded lists, not from invented hostnames.
Malicious examples are hagezi/nrd entropy DGA domains. Benign examples are
Majestic Million hostnames with any overlap removed.
"""

import os
import random
import re

# Character classes accepted by the tokenizer in train.py / agent.py.
# Must stay in sync with VALID_CHARS there.
DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://")

# Model input length used by src/train.py and src/agent.py.
MODEL_MAX_LEN = 45


def normalize_domain(value):
    """Return a lowercase hostname, or None when the value is not a domain.

    Strips comments, schemes, credentials, ports, paths, queries, and
    fragments. Internationalized names are converted to punycode. IP
    addresses and wildcard names are rejected.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text.startswith("#") or text.startswith(";"):
        return None

    text = _SCHEME_RE.sub("", text)
    if "@" in text:
        text = text.split("@", 1)[1]
    text = text.split("/", 1)[0]
    text = text.split("?", 1)[0]
    text = text.split("#", 1)[0]
    if text.startswith("[") and "]" in text:
        return None
    if ":" in text:
        text = text.split(":", 1)[0]
    text = text.strip().strip(".")
    if not text or "*" in text or " " in text:
        return None

    if any(ord(char) > 127 for char in text):
        try:
            text = text.encode("idna").decode("ascii")
        except (UnicodeError, UnicodeDecodeError):
            return None

    if _IPV4_RE.match(text) or not DOMAIN_RE.match(text):
        return None
    return text


def parse_domain_lines(lines):
    """Parse a comment-header domain list.

    Returns (metadata, domains) where metadata maps header keys to values
    and domains is a set of normalized hostnames.
    """
    metadata = {}
    domains = set()
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("#"):
            body = raw[1:].strip()
            if ":" in body:
                key, val = body.split(":", 1)
                metadata[key.strip().lower()] = val.strip()
            continue
        domain = normalize_domain(raw)
        if domain:
            domains.add(domain)
    return metadata, domains


def parse_majestic_csv(lines):
    """Parse the Majestic Million CSV and return (row_count, domains)."""
    import csv

    reader = csv.reader(lines)
    header = next(reader, None)
    if not header:
        return 0, set()
    try:
        domain_idx = [col.strip().lower() for col in header].index("domain")
    except ValueError:
        domain_idx = 2

    rows = 0
    domains = set()
    for row in reader:
        if len(row) <= domain_idx:
            continue
        rows += 1
        domain = normalize_domain(row[domain_idx])
        if domain:
            domains.add(domain)
    return rows, domains


def read_domain_list(path):
    """Load a one-domain-per-line file, normalizing and de-duplicating."""
    domains = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            domain = normalize_domain(line)
            if domain and domain not in seen:
                seen.add(domain)
                domains.append(domain)
    return domains


def load_labeled_domains(data_dir):
    """Load benign (0) and malicious (1) lists from a data directory."""
    benign_path = os.path.join(data_dir, "benign_domains.txt")
    malicious_path = os.path.join(data_dir, "malicious_domains.txt")
    missing = [path for path in (benign_path, malicious_path) if not os.path.isfile(path)]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing training lists: {joined}. "
            "Download them with: python src/download_datasets.py"
        )
    benign = read_domain_list(benign_path)
    malicious = read_domain_list(malicious_path)
    domains = benign + malicious
    labels = [0] * len(benign) + [1] * len(malicious)
    return domains, labels


def _sample(items, count, rng):
    """Deterministic sample. `items` must already be in a stable order."""
    if count >= len(items):
        return list(items)
    return rng.sample(list(items), count)


def build_training_lists(window_domains, benign_domains, max_per_class=0, seed=42, primary_window="7"):
    """Assemble equal benign and malicious lists from downloaded pools.

    The primary window (hagezi 7-day list) is kept in full when it fits in
    the quota. Older windows fill the remaining malicious slots. Any
    hostname that appears in any malicious window is removed from benign
    before balancing, including names that are not selected for the sample.
    """
    if primary_window not in window_domains:
        raise ValueError(f"Primary window {primary_window!r} is missing")

    windows = {name: set(domains) for name, domains in window_domains.items()}
    primary = windows[primary_window]
    union = set()
    for domains in windows.values():
        union |= domains
    extra = union - primary

    benign = set(benign_domains) - union
    overlap = len(set(benign_domains) & union)

    quota = min(len(benign), len(union))
    if max_per_class and max_per_class > 0:
        quota = min(quota, int(max_per_class))
    if quota <= 0:
        raise ValueError("No domains left after normalization and overlap removal")

    rng = random.Random(seed)
    primary_sorted = sorted(primary)
    extra_sorted = sorted(extra)
    benign_sorted = sorted(benign)

    if quota <= len(primary_sorted):
        malicious = _sample(primary_sorted, quota, rng)
        from_primary = len(malicious)
        from_older = 0
    else:
        need = quota - len(primary_sorted)
        older = _sample(extra_sorted, need, rng)
        malicious = list(primary_sorted) + older
        from_primary = len(primary_sorted)
        from_older = len(older)

    benign_sample = _sample(benign_sorted, quota, rng)
    malicious.sort()
    benign_sample.sort()

    longer_than_model = sum(1 for domain in malicious if len(domain) > MODEL_MAX_LEN)
    longer_than_model += sum(1 for domain in benign_sample if len(domain) > MODEL_MAX_LEN)

    return {
        "benign": benign_sample,
        "malicious": malicious,
        "overlap_removed_from_benign": overlap,
        "malicious_from_primary": from_primary,
        "malicious_from_older_windows": from_older,
        "benign_pool": len(benign),
        "malicious_union": len(union),
        "longer_than_model_max_len": longer_than_model,
    }

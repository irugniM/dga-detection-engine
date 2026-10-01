"""Download hagezi entropy DGA lists and a balanced benign domain list.

Malicious source (GPL-3.0):
  https://github.com/hagezi/nrd
  7-day feed:  https://cdn.jsdelivr.net/gh/hagezi/nrd@latest/domains/dga7.txt
  14-day feed: https://cdn.jsdelivr.net/gh/hagezi/nrd@latest/domains/dga14.txt
  30-day feed: https://cdn.jsdelivr.net/gh/hagezi/nrd@latest/domains/dga30.txt

Benign source (CC BY 3.0):
  Majestic Million, https://downloads.majestic.com/majestic_million.csv
  https://majestic.com/reports/majestic-million

The 7-day list is the primary malicious feed and is kept in full when it
fits the balanced quota. The 14- and 30-day lists add older entropy
registrations up to that quota. Benign names that also appear on any
hagezi window are dropped so the same hostname is never both labels.

The optional 1275.ru family feed is not downloaded. That file does not
state a license, and this project trains a binary classifier.
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import requests

from domains import build_training_lists, parse_domain_lines, parse_majestic_csv

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")

HAGEZI_LISTS = {
    "7": "https://cdn.jsdelivr.net/gh/hagezi/nrd@latest/domains/dga7.txt",
    "14": "https://cdn.jsdelivr.net/gh/hagezi/nrd@latest/domains/dga14.txt",
    "30": "https://cdn.jsdelivr.net/gh/hagezi/nrd@latest/domains/dga30.txt",
}
HAGEZI_HOME = "https://github.com/hagezi/nrd"
HAGEZI_LICENSE = "https://github.com/hagezi/nrd/blob/main/LICENSE"

MAJESTIC_URL = "https://downloads.majestic.com/majestic_million.csv"
MAJESTIC_PAGE = "https://majestic.com/reports/majestic-million"
MAJESTIC_LICENSE = "https://creativecommons.org/licenses/by/3.0/"

USER_AGENT = "dga-detection-engine/dataset-refresh"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, dest):
    """Download `url` to `dest` with a few retries. Returns the response headers."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    partial = dest + ".part"
    last_error = None
    for attempt in range(3):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=120,
                headers={"User-Agent": USER_AGENT},
            ) as response:
                response.raise_for_status()
                with open(partial, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            handle.write(chunk)
                os.replace(partial, dest)
                return {
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_length": response.headers.get("Content-Length"),
                }
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def write_domains(path, domains):
    with open(path, "w", encoding="utf-8") as handle:
        for domain in domains:
            handle.write(domain)
            handle.write("\n")


def write_notice(path):
    text = """Training domain lists in this directory were prepared by src/download_datasets.py.
The domain lists are separate works from the detection-engine code.

Malicious domains
  Source: hagezi/nrd entropy DGA / newly registered entropy domain lists
  Homepage: https://github.com/hagezi/nrd
  Feeds: domains/dga7.txt, domains/dga14.txt, domains/dga30.txt
  Retrieved via jsDelivr:
    https://cdn.jsdelivr.net/gh/hagezi/nrd@latest/domains/dga7.txt
    https://cdn.jsdelivr.net/gh/hagezi/nrd@latest/domains/dga14.txt
    https://cdn.jsdelivr.net/gh/hagezi/nrd@latest/domains/dga30.txt
  License: GNU General Public License v3.0
  License text: https://github.com/hagezi/nrd/blob/main/LICENSE
  benign_domains.txt / malicious_domains.txt are normalized, de-duplicated
  subsets (comments removed, names lowercased, scheme and path stripped).
  Copyright in the hagezi lists remains with their authors. Those subsets
  stay under GPL-3.0.

Benign domains
  Source: Majestic Million
  Download: https://downloads.majestic.com/majestic_million.csv
  Description: https://majestic.com/reports/majestic-million
  License: Creative Commons Attribution 3.0 Unported (CC BY 3.0)
  License text: https://creativecommons.org/licenses/by/3.0/
  Credit: Majestic (https://majestic.com).
  Changes: the CSV was reduced to normalized hostnames, names that also
  appear on a hagezi entropy list were removed, and a deterministic sample
  was taken so the benign count matches the malicious count.

Not included
  https://1275.ru/DGA/dga.txt was not merged. The feed does not state a
  license, and family labels are unused by this binary classifier.
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def load_hagezi_window(name, url):
    dest = os.path.join(RAW_DIR, f"dga{name}.txt")
    print(f"[*] Downloading hagezi {name}-day entropy list...")
    headers = download(url, dest)
    with open(dest, "r", encoding="utf-8", errors="replace") as handle:
        metadata, domains = parse_domain_lines(handle)
    print(f"[+] {name}-day list: {len(domains)} normalized domains")
    return {
        "window_days": int(name),
        "url": url,
        "file": os.path.relpath(dest, os.path.join(DATA_DIR, "..")),
        "sha256": sha256_file(dest),
        "http_last_modified": headers.get("last_modified"),
        "header": metadata,
        "normalized_domains": len(domains),
        "domains": domains,
    }


def load_majestic():
    dest = os.path.join(RAW_DIR, "majestic_million.csv")
    print("[*] Downloading Majestic Million benign list...")
    headers = download(MAJESTIC_URL, dest)
    with open(dest, "r", encoding="utf-8", errors="replace", newline="") as handle:
        rows, domains = parse_majestic_csv(handle)
    print(f"[+] Majestic: {rows} CSV rows, {len(domains)} normalized domains")
    return {
        "url": MAJESTIC_URL,
        "page": MAJESTIC_PAGE,
        "license": "CC BY 3.0",
        "license_url": MAJESTIC_LICENSE,
        "file": os.path.relpath(dest, os.path.join(DATA_DIR, "..")),
        "sha256": sha256_file(dest),
        "http_last_modified": headers.get("last_modified"),
        "csv_rows": rows,
        "normalized_domains": len(domains),
        "domains": domains,
    }


def main():
    parser = argparse.ArgumentParser(description="Download balanced DGA training lists")
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Directory for benign_domains.txt, malicious_domains.txt, and provenance.json",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Cap each class at this many domains. 0 keeps the full balanced set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    global RAW_DIR
    RAW_DIR = os.path.join(data_dir, "raw")
    os.makedirs(RAW_DIR, exist_ok=True)

    windows = {}
    window_meta = {}
    for name, url in HAGEZI_LISTS.items():
        loaded = load_hagezi_window(name, url)
        windows[name] = loaded.pop("domains")
        window_meta[name] = loaded

    majestic = load_majestic()
    benign_pool = majestic.pop("domains")

    print("[*] Balancing labels (full 7-day list, then older windows)...")
    selected = build_training_lists(
        windows,
        benign_pool,
        max_per_class=args.max_per_class,
        seed=args.seed,
        primary_window="7",
    )

    benign_path = os.path.join(data_dir, "benign_domains.txt")
    malicious_path = os.path.join(data_dir, "malicious_domains.txt")
    write_domains(benign_path, selected["benign"])
    write_domains(malicious_path, selected["malicious"])
    write_notice(os.path.join(data_dir, "NOTICE"))

    union = set()
    for domains in windows.values():
        union |= domains
    only_outside_30 = len((windows["7"] | windows["14"]) - windows["30"])

    provenance = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "model_max_len": 45,
        "malicious": {
            "source": "hagezi/nrd",
            "homepage": HAGEZI_HOME,
            "license": "GPL-3.0",
            "license_url": HAGEZI_LICENSE,
            "description": (
                "Newly registered entropy domains classified by hagezi as "
                "entropy NRDs/DGAs. Labels are that list's classification, "
                "not a separate malware-family ground truth."
            ),
            "windows_used": ["7", "14", "30"],
            "windows": window_meta,
            "union_normalized": len(union),
            "domains_in_7_or_14_missing_from_30": only_outside_30,
            "sampling": (
                "The 7-day list is included in full when it fits the balanced "
                "quota. Remaining malicious slots are a deterministic sample "
                "of domains that appear only on the 14-day or 30-day lists."
            ),
            "selected": len(selected["malicious"]),
            "selected_from_7_day": selected["malicious_from_primary"],
            "selected_from_older_windows": selected["malicious_from_older_windows"],
        },
        "benign": {
            "source": "Majestic Million",
            "url": MAJESTIC_URL,
            "page": MAJESTIC_PAGE,
            "license": "CC BY 3.0",
            "license_url": MAJESTIC_LICENSE,
            "attribution": (
                "Benign domains from the Majestic Million "
                "(https://majestic.com/reports/majestic-million), "
                "licensed under CC BY 3.0. Credit: Majestic."
            ),
            "csv_rows": majestic["csv_rows"],
            "normalized_domains": majestic["normalized_domains"],
            "sha256": majestic["sha256"],
            "http_last_modified": majestic["http_last_modified"],
            "overlap_removed": selected["overlap_removed_from_benign"],
            "pool_after_overlap_removal": selected["benign_pool"],
            "selected": len(selected["benign"]),
        },
        "training_corpus": {
            "benign": len(selected["benign"]),
            "malicious": len(selected["malicious"]),
            "balance": "equal",
            "longer_than_model_max_len": selected["longer_than_model_max_len"],
            "benign_sha256": sha256_file(benign_path),
            "malicious_sha256": sha256_file(malicious_path),
        },
        "not_used": {
            "1275_ru": {
                "url": "https://1275.ru/DGA/dga.txt",
                "reason": (
                    "Not merged. The feed does not state a license, and the "
                    "classifier is binary so family labels are unused."
                ),
            }
        },
    }
    provenance_path = os.path.join(data_dir, "provenance.json")
    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    corpus = provenance["training_corpus"]
    print(
        f"[+] Wrote {corpus['benign']} benign and {corpus['malicious']} malicious domains"
    )
    print(f"[+] Provenance: {provenance_path}")


if __name__ == "__main__":
    main()

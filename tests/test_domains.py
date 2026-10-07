import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from domains import (
    build_training_lists,
    normalize_domain,
    parse_domain_lines,
    parse_majestic_csv,
    read_domain_list,
)


def test_normalize_domain_strips_noise():
    assert normalize_domain("  HTTPS://User:pw@Example.COM:8443/a/b?q=1#x  ") == "example.com"
    assert normalize_domain("WWW.Example.NET.") == "www.example.net"
    assert normalize_domain("# comment") is None
    assert normalize_domain("") is None
    assert normalize_domain("not a domain") is None
    assert normalize_domain("*.evil.com") is None
    assert normalize_domain("1.2.3.4") is None
    assert normalize_domain("http://127.0.0.1/admin") is None


def test_normalize_domain_punycode():
    assert normalize_domain("münchen.de") == "xn--mnchen-3ya.de"


def test_parse_hagezi_header_and_dedupe():
    text = """# Title: Newly registered entropy domains (entropy NRDs/DGAs)
# Version: 2026.1001.0630.05
# Number of entries: 3
#
Example.COM
https://example.com/path
not a domain
second.org
"""
    meta, domains = parse_domain_lines(text.splitlines())
    assert meta["version"] == "2026.1001.0630.05"
    assert domains == {"example.com", "second.org"}


def test_parse_majestic_csv():
    text = """GlobalRank,TldRank,Domain,TLD
1,1,Google.com,com
2,2,https://play.google.com/store,com
3,3,not a domain,com
"""
    rows, domains = parse_majestic_csv(text.splitlines())
    assert rows == 3
    assert domains == {"google.com", "play.google.com"}


def test_build_training_lists_prefers_primary_and_drops_overlap():
    windows = {
        "7": {"primary-a.com", "primary-b.com", "shared.com"},
        "14": {"primary-a.com", "older-a.com"},
        "30": {"older-a.com", "older-b.com", "shared.com"},
    }
    benign = {
        "google.com",
        "shared.com",
        "older-b.com",
        "benign-a.com",
        "benign-b.net",
        "benign-c.org",
    }
    selected = build_training_lists(windows, benign, max_per_class=4, seed=42, primary_window="7")

    assert len(selected["benign"]) == len(selected["malicious"]) == 4
    assert "shared.com" not in selected["benign"]
    assert "older-b.com" not in selected["benign"]
    # Quota is larger than the primary window, so every primary name is kept.
    assert {"primary-a.com", "primary-b.com", "shared.com"} <= set(selected["malicious"])
    assert selected["malicious_from_primary"] == 3
    assert selected["malicious_from_older_windows"] == 1
    assert selected["overlap_removed_from_benign"] == 2


def test_build_training_lists_is_deterministic():
    windows = {"7": {f"p{i}.com" for i in range(10)}, "14": set(), "30": {f"o{i}.com" for i in range(10)}}
    benign = {f"b{i}.org" for i in range(30)}
    first = build_training_lists(windows, benign, max_per_class=6, seed=42)
    second = build_training_lists(windows, benign, max_per_class=6, seed=42)
    assert first["benign"] == second["benign"]
    assert first["malicious"] == second["malicious"]


def test_read_domain_list_skips_comments(tmp_path):
    path = tmp_path / "domains.txt"
    path.write_text("# comment\n\nExample.COM\nexample.com\n", encoding="utf-8")
    assert read_domain_list(str(path)) == ["example.com"]


def test_build_training_lists_rejects_empty_pool():
    with pytest.raises(ValueError):
        build_training_lists({"7": {"evil.com"}}, {"evil.com"}, primary_window="7")

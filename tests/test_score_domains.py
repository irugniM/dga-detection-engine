import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import score_domains
from score_domains import (
    MAX_LEN,
    collect_domains,
    default_model_path,
    format_result,
    import_tflite_module,
    load_char_index,
    tokenize_domain,
)


VOCAB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "char_index.json")
)


def test_tokenize_pads_and_truncates_like_training():
    char_index = load_char_index(VOCAB_PATH)
    short = tokenize_domain("abc.com", char_index)

    assert short.shape == (1, MAX_LEN)
    assert short.dtype == np.int32
    assert list(short[0, :7]) == [
        char_index["a"],
        char_index["b"],
        char_index["c"],
        char_index["."],
        char_index["c"],
        char_index["o"],
        char_index["m"],
    ]
    assert np.all(short[0, 7:] == 0)

    long_domain = "a" * 100 + ".ru"
    long_tokens = tokenize_domain(long_domain, char_index)
    assert long_tokens.shape == (1, MAX_LEN)
    assert np.all(long_tokens[0, :] == char_index["a"])

    unknown = tokenize_domain("ab_c.com", char_index)
    assert unknown[0, 2] == 0
    assert unknown[0, 3] == char_index["c"]


def test_tokenize_strips_scheme_and_path():
    char_index = load_char_index(VOCAB_PATH)
    plain = tokenize_domain("google.com", char_index)
    url = tokenize_domain("HTTPS://Google.com/index.html", char_index)
    assert np.array_equal(plain, url)


def test_format_result_fraction_percent_and_verdict():
    assert format_result("google.com", 0.123456, 0.85) == "google.com  0.123456 (12.35%)  ok"
    assert format_result("evil.cc", 0.85, 0.85) == "evil.cc  0.850000 (85.00%)  ALERT"
    assert format_result("evil.cc", 0.849999, 0.85) == "evil.cc  0.849999 (85.00%)  ok"


def test_collect_domains_from_args_and_stdin():
    domains = collect_domains(
        [" google.com ", "# skip", ""],
        ["qwertyuiopasdfghjkl.cc\n", "\n", "# comment\n", "  other.test  \n"],
    )
    assert domains == ["google.com", "qwertyuiopasdfghjkl.cc", "other.test"]


def test_default_model_prefers_tflite_then_keras(monkeypatch, tmp_path):
    tflite = tmp_path / "dga_lstm_model.tflite"
    keras = tmp_path / "dga_lstm_model.keras"
    monkeypatch.setattr(score_domains, "TFLITE_PATH", str(tflite))
    monkeypatch.setattr(score_domains, "KERAS_PATH", str(keras))

    tflite.write_bytes(b"tflite")
    keras.write_bytes(b"keras")
    assert default_model_path() == str(tflite)

    tflite.unlink()
    assert default_model_path() == str(keras)


def test_missing_tflite_runtime_falls_back_to_keras(monkeypatch, tmp_path, capsys):
    tflite = tmp_path / "dga_lstm_model.tflite"
    keras = tmp_path / "dga_lstm_model.keras"
    tflite.write_bytes(b"tflite")
    keras.write_bytes(b"keras")

    def missing_runtime():
        raise ImportError("no runtime")

    monkeypatch.setattr(score_domains, "import_tflite_module", missing_runtime)

    created = {}

    class FakeKeras:
        def __init__(self, path):
            created["path"] = path

    monkeypatch.setattr(score_domains, "KerasBackend", FakeKeras)

    backend, loaded = score_domains.load_backend(str(tflite), allow_keras_fallback=True)
    captured = capsys.readouterr()

    assert isinstance(backend, FakeKeras)
    assert created["path"] == str(keras)
    assert loaded == str(keras)
    assert "falling back" in captured.err


def test_explicit_tflite_path_does_not_fall_back(monkeypatch, tmp_path):
    tflite = tmp_path / "dga_lstm_model.tflite"
    keras = tmp_path / "dga_lstm_model.keras"
    tflite.write_bytes(b"tflite")
    keras.write_bytes(b"keras")

    def missing_runtime():
        raise ImportError("no runtime")

    monkeypatch.setattr(score_domains, "import_tflite_module", missing_runtime)

    with pytest.raises(ImportError, match="no runtime"):
        score_domains.load_backend(str(tflite), allow_keras_fallback=False)


def test_tflite_import_does_not_load_tensorflow(monkeypatch):
    imported = []

    def tracking_import(name, package=None):
        imported.append(name)
        raise ImportError(name)

    monkeypatch.setattr(score_domains.importlib, "import_module", tracking_import)
    with pytest.raises(ImportError, match="ai-edge-litert or tflite-runtime"):
        import_tflite_module()

    assert imported == list(score_domains.TFLITE_MODULES)
    assert not any(name.startswith("tensorflow") for name in imported)


def test_main_prints_scores_without_loading_a_real_model(monkeypatch, capsys):
    class FakeScorer:
        model_path = "models/dga_lstm_model.tflite"

        def score(self, domain):
            return {"google.com": 0.01, "qwertyuiopasdfghjkl.cc": 0.99}[domain]

    monkeypatch.setattr(score_domains, "load_scorer", lambda *args, **kwargs: FakeScorer())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    code = score_domains.main(["google.com", "qwertyuiopasdfghjkl.cc"])
    captured = capsys.readouterr()

    assert code == 0
    assert "google.com  0.010000 (1.00%)  ok" in captured.out
    assert "qwertyuiopasdfghjkl.cc  0.990000 (99.00%)  ALERT" in captured.out
    assert "threshold: 0.85" in captured.err


def test_main_rejects_empty_input(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    code = score_domains.main([])
    captured = capsys.readouterr()
    assert code == 2
    assert "No domain names" in captured.err

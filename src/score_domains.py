"""Print DGA model confidence for domain names without running the live agent.

Prefers ``models/dga_lstm_model.tflite`` via ai-edge-litert or tflite-runtime.
If that file or a lightweight runtime is missing, falls back to
``models/dga_lstm_model.keras`` and only then imports TensorFlow.
Tokenization matches ``train.py`` and ``agent.py``: lowercase, unknown
characters map to 0, and sequences are post-padded or truncated to MAX_LEN.
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np

MAX_LEN = 45
DEFAULT_THRESHOLD = 0.85

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
TFLITE_PATH = os.path.join(MODEL_DIR, "dga_lstm_model.tflite")
KERAS_PATH = os.path.join(MODEL_DIR, "dga_lstm_model.keras")
VOCAB_PATH = os.path.join(MODEL_DIR, "char_index.json")

# Lightweight interpreters only. TensorFlow is imported later, and only for .keras.
TFLITE_MODULES = (
    "ai_edge_litert.interpreter",
    "tflite_runtime.interpreter",
)


def tokenize_domain(domain, char_index):
    """Tokenize one domain the same way the trainer and the agent do.

    Lowercases, strips a scheme and path when present, maps each character
    through ``char_index`` (unknown characters become 0), then post-pads or
    truncates to ``MAX_LEN``.
    """
    domain = domain.lower().strip()
    # Same cleaning as agent.preprocess_domain: drop a scheme and any path.
    if "://" in domain:
        domain = domain.split("://")[-1]
    domain = domain.split("/")[0]

    tokens = [int(char_index.get(char, 0)) for char in domain]
    if len(tokens) < MAX_LEN:
        tokens.extend([0] * (MAX_LEN - len(tokens)))
    else:
        tokens = tokens[:MAX_LEN]
    return np.asarray([tokens], dtype=np.int32)


def format_result(domain, score, threshold):
    """One output line: domain, fraction, percent, and ALERT or ok."""
    verdict = "ALERT" if score >= threshold else "ok"
    return f"{domain}  {score:.6f} ({score * 100:.2f}%)  {verdict}"


def collect_domains(cli_domains, stdin_lines=None):
    """Gather domain names from CLI arguments and, when given, stdin lines."""
    domains = []
    for raw in cli_domains:
        domain = raw.strip()
        if domain and not domain.startswith("#"):
            domains.append(domain)
    if stdin_lines is not None:
        for line in stdin_lines:
            domain = line.strip()
            if domain and not domain.startswith("#"):
                domains.append(domain)
    return domains


def default_model_path():
    """Prefer the TFLite file when it exists, otherwise the Keras file."""
    if os.path.exists(TFLITE_PATH):
        return TFLITE_PATH
    if os.path.exists(KERAS_PATH):
        return KERAS_PATH
    return TFLITE_PATH


def load_char_index(path):
    with open(path, encoding="utf-8") as handle:
        char_index = json.load(handle)
    if not isinstance(char_index, dict):
        raise ValueError(f"Character index at {path} must be a JSON object.")
    return {str(key): int(value) for key, value in char_index.items()}


def import_tflite_module():
    """Import a lightweight TFLite interpreter module, never TensorFlow."""
    failures = []
    for module_name in TFLITE_MODULES:
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(f"{module_name} ({exc})")
    raise ImportError(
        "A .tflite model requires ai-edge-litert or tflite-runtime. "
        "Tried: " + "; ".join(failures) + ". "
        "Install one of those packages, or pass --model for the .keras file."
    )


class TfliteBackend:
    def __init__(self, module, model_path):
        self.interpreter = module.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, tokens):
        details = self.input_details[0]
        self.interpreter.set_tensor(details["index"], tokens.astype(details["dtype"]))
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return float(np.asarray(output).reshape(-1)[0])


class KerasBackend:
    def __init__(self, model_path):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "Loading a .keras model requires TensorFlow. "
                "Install tensorflow, or use the .tflite model with "
                "ai-edge-litert or tflite-runtime."
            ) from exc
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, tokens):
        prediction = self.model.predict(tokens, verbose=0)
        return float(np.asarray(prediction).reshape(-1)[0])


def load_backend(model_path, allow_keras_fallback):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.endswith(".tflite"):
        try:
            module = import_tflite_module()
        except ImportError:
            keras_path = os.path.splitext(model_path)[0] + ".keras"
            if allow_keras_fallback and os.path.exists(keras_path):
                print(
                    f"[*] No lightweight TFLite runtime; falling back to {keras_path}",
                    file=sys.stderr,
                )
                return KerasBackend(keras_path), keras_path
            raise
        return TfliteBackend(module, model_path), model_path

    return KerasBackend(model_path), model_path


class DomainScorer:
    def __init__(self, backend, char_index, model_path):
        self.backend = backend
        self.char_index = char_index
        self.model_path = model_path

    def score(self, domain):
        return self.backend.predict(tokenize_domain(domain, self.char_index))


def load_scorer(model_path, vocab_path, allow_keras_fallback):
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    char_index = load_char_index(vocab_path)
    backend, loaded_path = load_backend(model_path, allow_keras_fallback)
    return DomainScorer(backend, char_index, os.path.normpath(loaded_path))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Print DGA model confidence scores for domain names.",
        epilog=(
            "Examples:\n"
            "  python src/score_domains.py google.com qwertyuiopasdfghjkl.cc\n"
            "  python src/score_domains.py --threshold 0.5 < domains.txt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "domains",
        nargs="*",
        help="Domain names to score. Stdin is also read when it is not a terminal.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Confidence at or above this value prints ALERT (default: 0.85).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path. Default: dga_lstm_model.tflite if present, else .keras.",
    )
    parser.add_argument(
        "--vocab",
        default=VOCAB_PATH,
        help="Path to char_index.json (default: models/char_index.json).",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0.0 <= args.threshold <= 1.0:
        print("[-] --threshold must be between 0 and 1.", file=sys.stderr)
        return 2

    stdin_lines = None if sys.stdin.isatty() else sys.stdin
    domains = collect_domains(args.domains, stdin_lines)
    if not domains:
        print(
            "[-] No domain names given. Pass them as arguments or pipe one name per line.",
            file=sys.stderr,
        )
        return 2

    model_path = args.model if args.model else default_model_path()
    allow_keras_fallback = args.model is None
    try:
        scorer = load_scorer(model_path, args.vocab, allow_keras_fallback)
    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
        print(f"[-] {exc}", file=sys.stderr)
        return 1

    print(
        f"[*] Model: {scorer.model_path}  threshold: {args.threshold:.2f}",
        file=sys.stderr,
    )
    for domain in domains:
        score = scorer.score(domain)
        print(format_result(domain, score, args.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())

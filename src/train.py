import os
import json
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from domains import load_labeled_domains

# --- CONFIGURATION ---
MAX_LEN = 45
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "dga_lstm_model.keras")
CHAR_INDEX_PATH = os.path.join(MODEL_DIR, "char_index.json")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")

# Valid characters in domain names (excluding protocol and sub-paths)
VALID_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."

# --- SYNTHETIC DATA GENERATION ---
# Kept for unit tests of tokenizer shapes. The training entrypoint does not
# call this; it loads data/benign_domains.txt and data/malicious_domains.txt.
def generate_synthetic_data(num_samples=5000):
    """
    Generates a synthetic balanced dataset of benign and DGA domain names.
    - Benign domains represent typical English-like structures and common web services.
    - DGA domains mimic random alphanumeric, hex-based, or high-entropy patterns.
    """
    random.seed(42)
    np.random.seed(42)

    benign_prefixes = ["google", "facebook", "youtube", "yahoo", "amazon", "wikipedia", "twitter", 
                       "linkedin", "instagram", "netflix", "reddit", "microsoft", "apple", "github", 
                       "stackoverflow", "medium", "spotify", "pinterest", "tumblr", "paypal", "ebay",
                       "craigslist", "dropbox", "vimeo", "wordpress", "blogger", "flickr", "imdb"]
    
    syllables = ["ba", "co", "da", "fe", "go", "ha", "ki", "lo", "ma", "ne", "pa", "ro", "si", "te", 
                 "un", "vi", "wa", "za", "ber", "lin", "ton", "gard", "field", "port", "land", "wood"]

    tlds = [".com", ".net", ".org", ".info", ".biz", ".us", ".uk", ".de", ".ru", ".cn", ".jp"]

    domains = []
    labels = []

    # 1. Generate Benign Domains
    for _ in range(num_samples // 2):
        rand_val = random.random()
        # Format A: Random combination of realistic syllables (e.g., "copaland.com")
        if rand_val < 0.40:
            parts = [random.choice(syllables) for _ in range(random.randint(2, 4))]
            domain = "".join(parts) + random.choice(tlds)
        # Format B: Brand prefix + suffix or hyphenated words (e.g., "google-support.net")
        elif rand_val < 0.75:
            base = random.choice(benign_prefixes)
            suffix = random.choice(syllables)
            domain = f"{base}-{suffix}{random.choice(tlds)}"
        # Format C: High-entropy looking CDN/cloud infrastructure subdomains (e.g., "a1024-xyz.akamai.net")
        else:
            infra_domains = [
                "akamai.net", "amazonaws.com", "cloudfront.net", "nordcdn.com", 
                "cursor.sh", "githubusercontent.com", "senecacollege.ca", "office365.com"
            ]
            selected_infra = random.choice(infra_domains)
            # Generate alphanumeric high-entropy looking subdomain
            length = random.randint(6, 15)
            sub = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789-") for _ in range(length))
            # Make sure it doesn't start or end with a hyphen
            sub = sub.strip("-")
            if not sub:
                sub = "cdn"
            domain = f"{sub}.{selected_infra}"
        
        # Add random subdomains occasionally to formats A & B
        if rand_val < 0.75 and random.random() < 0.15:
            domain = random.choice(["www", "api", "mail", "blog"]) + "." + domain
            
        domains.append(domain)
        labels.append(0) # 0 = Benign

    # 2. Generate DGA (Malicious) Domains
    dga_tlds = [".ru", ".xyz", ".cc", ".su", ".click", ".top", ".info", ".biz"]
    for _ in range(num_samples // 2):
        dga_type = random.random()
        
        # Type A: High-entropy random alphanumeric string (e.g., "sfg94ka91vxa.ru")
        if dga_type < 0.5:
            length = random.randint(10, 25)
            chars = [random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(length)]
            domain = "".join(chars) + random.choice(dga_tlds)
        # Type B: Random hex-based or consonant heavy strings (e.g., "xqtzpwkdbm.xyz")
        elif dga_type < 0.8:
            length = random.randint(8, 18)
            chars = [random.choice("bcdfghjklmnpqrstvwxyz0123456789") for _ in range(length)]
            domain = "".join(chars) + random.choice(dga_tlds)
        # Type C: Dictionary/syllable collision but with high repetition or lengths (e.g., "babazazaneviwa.click")
        else:
            parts = [random.choice(syllables) for _ in range(random.randint(5, 8))]
            domain = "".join(parts) + random.choice(dga_tlds)
            
        domains.append(domain)
        labels.append(1) # 1 = DGA/Malicious

    return domains, labels

# --- PREPROCESSING ---
def create_vocab():
    """Creates a character-to-index mapping for tokenization."""
    # 0 is reserved for padding
    char_index = {char: idx + 1 for idx, char in enumerate(VALID_CHARS)}
    return char_index

def tokenize_and_pad(domains, char_index):
    """Converts a list of domain strings to padded index sequences."""
    tokenized = np.zeros((len(domains), MAX_LEN), dtype=np.int32)
    for row, domain in enumerate(domains):
        domain = domain.lower().strip()
        # Convert chars to indices; default to 0 for unknown chars.
        # Sequences longer than MAX_LEN are truncated.
        limit = min(len(domain), MAX_LEN)
        for col in range(limit):
            tokenized[row, col] = char_index.get(domain[col], 0)
    return tokenized

# --- MODEL ARCHITECTURE ---
def build_lstm_model(vocab_size):
    """Builds a high-accuracy CNN-LSTM hybrid network using Keras Sequential API."""
    model = tf.keras.Sequential([
        # Input Layer: sequence length = MAX_LEN
        tf.keras.layers.Input(shape=(MAX_LEN,)),
        
        # Embedding Layer: Maps tokens to a 32-dimensional dense space
        tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=32),
        
        # 1D Convolutional Layer: Extracts local character clusters and n-gram structures
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Bidirectional LSTM Layer: Models sequential character ordering forwards and backwards
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        
        # Dropout: Regularization to prevent overfitting
        tf.keras.layers.Dropout(0.5),
        
        # Dense Layer: Fully connected decision features
        tf.keras.layers.Dense(32, activation='relu'),
        
        # Output Layer: Sigmoid activation for binary probability
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

# --- MAIN EXECUTION ---
def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    print("[*] Setting up directories...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("[*] Generating vocabulary mapping...")
    char_index = create_vocab()
    with open(CHAR_INDEX_PATH, 'w') as f:
        json.dump(char_index, f, indent=4)
    print(f"[+] Saved token vocabulary to {CHAR_INDEX_PATH}")

    print("[*] Loading downloaded domain lists...")
    try:
        domains, labels = load_labeled_domains(DATA_DIR)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    y = np.array(labels)
    n_benign = int(np.sum(y == 0))
    n_malicious = int(np.sum(y == 1))
    print(f"[+] Loaded {len(domains)} domains ({n_benign} benign, {n_malicious} malicious)")

    print("[*] Preprocessing and padding data...")
    X = tokenize_and_pad(domains, char_index)

    # Train / Test Split (80% Train, 20% Test), stratified so the test set stays balanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[+] Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    print("[*] Initializing Bidirectional LSTM Neural Network...")
    vocab_size = len(char_index)
    model = build_lstm_model(vocab_size)
    model.summary()

    print("[*] Training DGA classifier model...")
    # Train for 5 epochs with batch size of 64
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=64,
        verbose=1
    )

    print("[*] Evaluating trained model on unseen test dataset...")
    loss, keras_accuracy, keras_precision, keras_recall = model.evaluate(X_test, y_test, verbose=0)

    # Positive class is malicious (label 1).
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n[+] Test Results:")
    print(f"    Loss:      {loss:.4f}")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f} (malicious)")
    print(f"    Recall:    {recall:.4f} (malicious)")
    print(f"    F1:        {f1:.4f} (malicious)")
    print(f"    ROC-AUC:   {roc_auc:.4f}")

    print("\n[+] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "DGA"]))
    print(f"[+] ROC-AUC Score: {roc_auc:.4f}\n")

    metrics = {
        "accuracy": round(float(accuracy), 6),
        "precision_malicious": round(float(precision), 6),
        "recall_malicious": round(float(recall), 6),
        "f1_malicious": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "loss": round(float(loss), 6),
        "keras_evaluate": {
            "accuracy": round(float(keras_accuracy), 6),
            "precision": round(float(keras_precision), 6),
            "recall": round(float(keras_recall), 6),
        },
        "dataset": {
            "total": int(len(y)),
            "benign": n_benign,
            "malicious": n_malicious,
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "test_benign": int(np.sum(y_test == 0)),
            "test_malicious": int(np.sum(y_test == 1)),
        },
        "threshold": 0.5,
        "epochs": 5,
        "batch_size": 64,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(f"[+] Saved metrics to {METRICS_PATH}")

    print(f"[*] Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("[+] Model saved successfully!")

    # --- TFLITE CONVERSION ---
    print("[*] Converting Keras model to TensorFlow Lite for ultra-lightweight edge deployment...")
    try:
        tflite_path = os.path.join(MODEL_DIR, "dga_lstm_model.tflite")
        
        # LSTMs require a concrete function with static shapes to avoid TF dynamic tensor list ops during conversion.
        run_model = tf.function(lambda x: model(x))
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec([1, MAX_LEN], model.inputs[0].dtype)
        )
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Use standard float32 model to maintain maximum backward compatibility with older Pi runtimes (avoids FULLY_CONNECTED v12 errors)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Standard built-in ops are sufficient for standard LSTMs when using static shape concrete functions
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"[+] TensorFlow Lite model saved to {tflite_path}")
    except Exception as e:
        print(f"[-] TensorFlow Lite conversion failed: {e}")

if __name__ == "__main__":
    main()

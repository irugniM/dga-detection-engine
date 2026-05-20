import os
import json
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# --- CONFIGURATION ---
MAX_LEN = 45
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "dga_lstm_model.keras")
CHAR_INDEX_PATH = os.path.join(MODEL_DIR, "char_index.json")

# Valid characters in domain names (excluding protocol and sub-paths)
VALID_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."

# --- SYNTHETIC DATA GENERATION ---
# This simulates a high-quality DGA and benign dataset for demonstration and training.
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
        # Format A: Random combination of realistic syllables (e.g., "copaland.com")
        if random.random() < 0.6:
            parts = [random.choice(syllables) for _ in range(random.randint(2, 4))]
            domain = "".join(parts) + random.choice(tlds)
        # Format B: Brand prefix + suffix or hyphenated words (e.g., "google-support.net")
        else:
            base = random.choice(benign_prefixes)
            suffix = random.choice(syllables)
            domain = f"{base}-{suffix}{random.choice(tlds)}"
        
        # Add random subdomains occasionally
        if random.random() < 0.15:
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
    tokenized_domains = []
    for domain in domains:
        domain = domain.lower().strip()
        # Convert chars to indices; default to 0 for unknown chars
        tokens = [char_index.get(char, 0) for char in domain]
        
        # Enforce static length (MAX_LEN) using post-padding or truncation
        if len(tokens) < MAX_LEN:
            tokens = tokens + [0] * (MAX_LEN - len(tokens))
        else:
            tokens = tokens[:MAX_LEN]
            
        tokenized_domains.append(tokens)
        
    return np.array(tokenized_domains)

# --- MODEL ARCHITECTURE ---
def build_lstm_model(vocab_size):
    """Builds a Bidirectional LSTM network using Keras Sequential API."""
    model = tf.keras.Sequential([
        # Input Layer: sequence length = MAX_LEN
        tf.keras.layers.Input(shape=(MAX_LEN,)),
        
        # Embedding Layer: Maps tokens to a 32-dimensional dense space
        tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=32, input_length=MAX_LEN),
        
        # Bidirectional LSTM Layer: Learns temporal character features forwards and backwards
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
    print("[*] Setting up directories...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("[*] Generating vocabulary mapping...")
    char_index = create_vocab()
    with open(CHAR_INDEX_PATH, 'w') as f:
        json.dump(char_index, f, indent=4)
    print(f"[+] Saved token vocabulary to {CHAR_INDEX_PATH}")

    print("[*] Synthesizing dataset for training...")
    domains, labels = generate_synthetic_data(num_samples=8000)
    print(f"[+] Synthesized {len(domains)} records (50% Benign, 50% DGA)")

    print("[*] Preprocessing and padding data...")
    X = tokenize_and_pad(domains, char_index)
    y = np.array(labels)

    # Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[+] Test Results:")
    print(f"    Loss:      {loss:.4f}")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")

    # Advanced Metrics
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    print("\n[+] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "DGA"]))
    
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"[+] ROC-AUC Score: {roc_auc:.4f}\n")

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

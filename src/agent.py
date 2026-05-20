import os
import sys
import json
import time
import random
import argparse
import threading
import numpy as np
import requests
import subprocess
from suricata_socket import SuricataSocketConnector

# Configure standard streams to use UTF-8 for cross-platform emoji support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def load_env_file(env_path):
    """Loads environment variables from a .env file if it exists."""
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip('"').strip("'")
        except Exception as e:
            print(f"[*] Warning: Failed to parse .env file: {e}")

# Paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load env file dynamically on startup
load_env_file(os.path.join(SCRIPT_DIR, "..", ".env"))

# --- DEFAULT CONFIGURATIONS ---
DEFAULT_THRESHOLD = 0.85
MAX_LEN = 45

# Default to .tflite if it exists, otherwise fallback to .keras
DEFAULT_TFLITE_PATH = os.path.join(SCRIPT_DIR, "..", "models", "dga_lstm_model.tflite")
DEFAULT_KERAS_PATH = os.path.join(SCRIPT_DIR, "..", "models", "dga_lstm_model.keras")
DEFAULT_MODEL_PATH = DEFAULT_TFLITE_PATH if os.path.exists(DEFAULT_TFLITE_PATH) else DEFAULT_KERAS_PATH
DEFAULT_VOCAB_PATH = os.path.join(SCRIPT_DIR, "..", "models", "char_index.json")

# Linux Production Paths
PIHOLE_LOG = "/var/log/pihole/pihole.log"
PIHOLE_CUSTOM_LIST = "/etc/pihole/custom.list"
SURICATA_RULES = "/etc/suricata/rules/local.rules"
SURICATA_SOCKET = "/var/run/suricata/suricata-command.socket"

# Telegram placeholders
DEFAULT_TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
DEFAULT_TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

class DGADetectorAgent:
    def __init__(self, args):
        self.mock = args.mock
        self.threshold = args.threshold
        # Resolve Telegram configurations, prioritizing CLI arguments unless they are placeholders
        self.telegram_token = args.telegram_token
        if self.telegram_token == DEFAULT_TELEGRAM_TOKEN:
            self.telegram_token = os.environ.get("TELEGRAM_TOKEN", os.environ.get("TOKEN", DEFAULT_TELEGRAM_TOKEN))

        self.telegram_chat_id = args.telegram_chat_id
        if self.telegram_chat_id == DEFAULT_TELEGRAM_CHAT_ID:
            self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", os.environ.get("CHAT_ID", DEFAULT_TELEGRAM_CHAT_ID))
        
        # Set paths depending on execution mode
        if self.mock:
            print("[*] Running in MOCK/DEVELOPMENT mode...")
            mock_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "mock_env"))
            os.makedirs(mock_dir, exist_ok=True)
            self.pihole_log = os.path.join(mock_dir, "pihole.log")
            self.pihole_custom_list = os.path.join(mock_dir, "custom.list")
            self.suricata_rules = os.path.join(mock_dir, "local.rules")
            # Touch files to ensure they exist
            open(self.pihole_log, "w").close()
            if not os.path.exists(self.pihole_custom_list):
                open(self.pihole_custom_list, "w").close()
            if not os.path.exists(self.suricata_rules):
                open(self.suricata_rules, "w").close()
        else:
            print("[*] Running in Linux PRODUCTION mode...")
            self.pihole_log = args.log_path
            self.pihole_custom_list = args.custom_list
            self.suricata_rules = args.rules_path

        # Setup Suricata connector
        self.suricata_connector = SuricataSocketConnector(
            socket_path=args.socket_path, 
            mock=self.mock
        )

        # Load Vocabulary mapping
        print(f"[*] Loading vocabulary index from {args.vocab}...")
        try:
            with open(args.vocab, 'r') as f:
                self.char_index = json.load(f)
            print(f"[+] Loaded vocabulary with {len(self.char_index)} characters.")
        except FileNotFoundError:
            print(f"[-] Vocabulary file not found at {args.vocab}. Please run training first!")
            sys.exit(1)

        # Load trained deep learning model (TFLite or Keras)
        print(f"[*] Loading model from {args.model}...")
        self.is_tflite = args.model.endswith(".tflite")

        if self.is_tflite:
            try:
                # Try LiteRT (modern Google TFLite runtime package) first
                import ai_edge_litert.interpreter as tflite
            except ImportError:
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    try:
                        import tensorflow.lite as tflite
                    except ImportError:
                        print("[-] Error: Running a .tflite model requires 'ai-edge-litert', 'tflite-runtime', or 'tensorflow' to be installed.")
                        print("    For a minimal Raspberry Pi setup, run: pip install ai-edge-litert")
                        sys.exit(1)
            
            try:
                self.interpreter = tflite.Interpreter(model_path=args.model)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("[+] TensorFlow Lite model loaded and allocated successfully.")
            except Exception as e:
                print(f"[-] Failed to load TensorFlow Lite model: {e}")
                sys.exit(1)
        else:
            # Dynamically import tensorflow only when loading Keras models
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(args.model)
                print("[+] Keras deep learning model loaded successfully.")
            except ImportError:
                print("[-] Error: 'tensorflow' is not installed. To load .keras models, please install tensorflow.")
                print("    Or specify a .tflite model path instead for lightweight tflite-runtime mode.")
                sys.exit(1)
            except Exception as e:
                print(f"[-] Failed to load Keras model: {e}")
                sys.exit(1)

        # Dynamic Rule Sid tracking (Read highest sid from file or start at base)
        self.current_sid = self._get_starting_sid()

    def _get_starting_sid(self) -> int:
        """Parses current rules file to find the largest SID to prevent collisions."""
        base_sid = 1000001
        if not os.path.exists(self.suricata_rules):
            return base_sid
        
        try:
            with open(self.suricata_rules, "r") as f:
                lines = f.readlines()
            for line in lines:
                if "sid:" in line:
                    # Extract sid value from rule format "sid:XXXXX;"
                    parts = line.split("sid:")
                    if len(parts) > 1:
                        sid_val = parts[1].split(";")[0].strip()
                        if sid_val.isdigit():
                            base_sid = max(base_sid, int(sid_val) + 1)
        except Exception as e:
            print(f"[-] Error scanning local.rules for SID base: {e}")
        return base_sid

    def preprocess_domain(self, domain: str) -> np.ndarray:
        """Tokenizes and post-pads the domain name to MAX_LEN for model compatibility."""
        domain = domain.lower().strip()
        # Clean potential URL schemas if logs capture full paths
        if "://" in domain:
            domain = domain.split("://")[-1]
        domain = domain.split("/")[0]

        tokens = [self.char_index.get(char, 0) for char in domain]
        
        if len(tokens) < MAX_LEN:
            tokens = tokens + [0] * (MAX_LEN - len(tokens))
        else:
            tokens = tokens[:MAX_LEN]
            
        return np.array([tokens])

    def trigger_defenses(self, domain: str, score: float):
        """Orchestrates Pi-hole DNS sinkhole, Suricata packet drop, and Telegram notification."""
        print(f"\n[!] TRIPPING NETWORK DEFENSES FOR MALICIOUS DOMAIN: {domain} (Score: {score:.4f})")
        
        # 1. Update Pi-hole custom list (local sinkhole DNS entry)
        self._block_on_pihole(domain)

        # 2. Add Suricata dynamic rule
        self._block_on_suricata(domain)

        # 3. Dispatches alert notification
        self._dispatch_alerts(domain, score)

    def _block_on_pihole(self, domain: str):
        """Appends domain to Pi-hole local custom blocklist and reloads dnsmasq."""
        # Avoid duplicate blocklist entries
        try:
            is_already_blocked = False
            if os.path.exists(self.pihole_custom_list):
                with open(self.pihole_custom_list, "r") as f:
                    content = f.read()
                if domain in content:
                    is_already_blocked = True

            if not is_already_blocked:
                with open(self.pihole_custom_list, "a") as f:
                    f.write(f"0.0.0.0 {domain}\n")
                print(f"[+] Appended '{domain}' to Pi-hole custom list: {self.pihole_custom_list}")
            else:
                print(f"[*] '{domain}' is already present in Pi-hole blocklist.")

            # Trigger reloading the DNS system
            if self.mock:
                print("[Mock] Reloading Pi-hole DNS subsystem: 'pihole restartdns reload'")
            else:
                subprocess.run(["pihole", "restartdns", "reload"], check=True, capture_output=True)
                print("[+] Pi-hole DNS reloaded successfully.")
        except Exception as e:
            print(f"[-] Failed to update Pi-hole custom list: {e}")

    def _block_on_suricata(self, domain: str):
        """Appends a packet dropping DNS signature to local rules and triggers reload."""
        suricata_rule = (
            f'drop dns any any -> any any (msg:"LSTM DGA Blocked: {domain}"; '
            f'dns.query; content:"{domain}"; sid:{self.current_sid}; rev:1;)\n'
        )
        try:
            # Check for duplicates in rules
            is_already_blocked = False
            if os.path.exists(self.suricata_rules):
                with open(self.suricata_rules, "r") as f:
                    content = f.read()
                if f'content:"{domain}"' in content:
                    is_already_blocked = True

            if not is_already_blocked:
                with open(self.suricata_rules, "a") as rules_file:
                    rules_file.write(suricata_rule)
                print(f"[+] Appended Suricata drop rule with SID {self.current_sid} to {self.suricata_rules}")
                self.current_sid += 1
            else:
                print(f"[*] Suricata rule for '{domain}' already exists.")

            # Reload Suricata via the high performance Unix socket
            self.suricata_connector.reload_ruleset()
            
        except Exception as e:
            print(f"[-] Failed to update Suricata rules: {e}")

    def _dispatch_alerts(self, domain: str, score: float):
        """Dispatches threat detections to Telegram channels."""
        message = f"🚨 *Malicious Indicator Detected!*\n*Domain:* `{domain}`\n*LSTM DGA Confidence Score:* `{score:.4f}`\n*Actions:* Blocked on Pi-hole & Suricata IDS"
        print(f"[Alert] Sending Telegram notification...")

        # If placeholders are present, print and exit gracefully to facilitate testing
        if self.telegram_token == DEFAULT_TELEGRAM_TOKEN or self.telegram_chat_id == DEFAULT_TELEGRAM_CHAT_ID:
            print(f"[Mock Notification] Token/Chat ID unconfigured. Alert Details:\n{message}")
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                print("[+] Telegram threat alert dispatched successfully.")
            else:
                print(f"[-] Telegram dispatch failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[-] Telegram network connection failed: {e}")

    def parse_dns_log_line(self, line: str) -> str:
        """Parses standard dnsmasq/Pi-hole syslog records for domain queries."""
        # Typical log format: "query[A] google.com from 192.168.1.100"
        if "query[" in line and "from" in line:
            parts = line.strip().split()
            try:
                # Find index containing query tag to handle varying syslog prefix lengths
                query_idx = next(i for i, part in enumerate(parts) if "query[" in part)
                if len(parts) > query_idx + 2 and parts[query_idx + 2] == "from":
                    domain = parts[query_idx + 1]
                    # Strip absolute TLD trailing dots
                    return domain.strip(".")
            except (StopIteration, IndexError):
                pass
        return None

    def run_inference_on_domain(self, domain: str) -> float:
        """Runs the processed domain through the LSTM model to compute score."""
        processed = self.preprocess_domain(domain)
        
        if self.is_tflite:
            input_type = self.input_details[0]['dtype']
            self.interpreter.set_tensor(self.input_details[0]['index'], processed.astype(input_type))
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        else:
            prediction = self.model.predict(processed, verbose=0)[0][0]
            
        return float(prediction)

    def listen(self):
        """Asynchronously tails the DNS query log and feeds findings to the LSTM."""
        print(f"[*] Starting DGA tail agent on log file: {self.pihole_log}...")
        
        if not os.path.exists(self.pihole_log):
            print(f"[-] Error: Target log file does not exist: {self.pihole_log}")
            sys.exit(1)

        # Open file and go to end of file
        with open(self.pihole_log, "r") as file_handle:
            file_handle.seek(0, 2) # Go to the end of the file
            print("[+] Inference engine listening and tailing live queries. Ready.")
            
            while True:
                line = file_handle.readline()
                if not line:
                    time.sleep(0.1) # Sleep briefly when no data is available
                    continue
                
                domain = self.parse_dns_log_line(line)
                if domain:
                    score = self.run_inference_on_domain(domain)
                    print(f"[*] Scored: {domain} -> {score:.4f}")
                    
                    if score >= self.threshold:
                        self.trigger_defenses(domain, score)


# --- MOCK EVENT GENERATOR THREAD ---
def generate_mock_log_traffic(log_file_path):
    """Simulates real-time system log additions for development validation."""
    print("[Mock System] Starting background traffic generator thread...")
    benign_pool = [
        "google.com", "github.com", "stackoverflow.com", "python.org", "wikipedia.org", 
        "apple.com", "microsoft.com", "openai.com", "weather.com", "cnn.com"
    ]
    dga_pool = [
        "xzrt883wpaq1.ru", "mnbqwsplk12.cc", "7788asdffgh12.su", "hjgffda091as2.biz", 
        "wqzx12300pa.click", "amazon-security-alert-login.info"
    ]

    time.sleep(2) # Give the tailer time to initialize
    counter = 0

    while True:
        # 1. Select a query (90% Benign, 10% DGA)
        if random.random() < 0.85:
            domain = random.choice(benign_pool)
        else:
            domain = random.choice(dga_pool)

        # 2. Build dnsmasq format log entry
        timestamp = time.strftime("%b %d %H:%M:%S")
        log_line = f"{timestamp} dnsmasq[1234]: query[A] {domain} from 192.168.1.150\n"
        
        # 3. Write line to mock log file
        try:
            with open(log_file_path, "a") as f:
                f.write(log_line)
        except Exception as e:
            print(f"[-] Mock Generator failed to write to log: {e}")
            break
            
        counter += 1
        time.sleep(random.uniform(1.5, 4.0))


# --- CLI ENTRYPOINT ---
def main():
    parser = argparse.ArgumentParser(description="LSTM-based Real-time DGA & Malicious Domain Detection Engine")
    parser.add_argument("--mock", action="store_true", help="Launch in mock/development mode for testing (no Linux sudo required)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Malicious confidence classification threshold (0.0 to 1.0)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained LSTM .keras model file")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB_PATH, help="Path to character index vocabulary .json file")
    parser.add_argument("--log-path", type=str, default=PIHOLE_LOG, help="Path to Pi-hole syslog file to tail")
    parser.add_argument("--custom-list", type=str, default=PIHOLE_CUSTOM_LIST, help="Path to Pi-hole custom list block file")
    parser.add_argument("--rules-path", type=str, default=SURICATA_RULES, help="Path to Suricata local.rules signature file")
    parser.add_argument("--socket-path", type=str, default=SURICATA_SOCKET, help="Path to Suricata Unix Domain Socket")
    parser.add_argument("--telegram-token", type=str, default=DEFAULT_TELEGRAM_TOKEN, help="Telegram Bot API Token")
    parser.add_argument("--telegram-chat_id", type=str, default=DEFAULT_TELEGRAM_CHAT_ID, help="Telegram Channel/Group Chat ID")
    
    args = parser.parse_args()

    # Create and run agent
    agent = DGADetectorAgent(args)

    # In mock mode, spin up the background traffic simulator
    if args.mock:
        traffic_thread = threading.Thread(
            target=generate_mock_log_traffic, 
            args=(agent.pihole_log,), 
            daemon=True
        )
        traffic_thread.start()

    # Enter the main listening log-tail loop
    try:
        agent.listen()
    except KeyboardInterrupt:
        print("\n[*] Exiting DGA Detection Agent daemon gracefully. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()

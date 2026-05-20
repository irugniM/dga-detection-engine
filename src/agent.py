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

        # Resolve Auto-Block configuration
        self.auto_block = getattr(args, "auto_block", None)
        if self.auto_block is None:
            env_val = os.environ.get("AUTO_BLOCK", "true").lower()
            self.auto_block = env_val not in ("false", "0", "no")

        # Built-in Whitelist of top/common legitimate infrastructure domains to prevent false positives
        self.builtin_whitelist = {
            "google.com", "youtube.com", "microsoft.com", "windows.com", 
            "windowsupdate.com", "live.com", "office.com", "office365.com", 
            "apple.com", "icloud.com", "github.com", "githubusercontent.com", 
            "amazon.com", "amazonaws.com", "cloudflare.com", "netflix.com", 
            "akamai.net", "akamaihd.net", "cursor.sh", "nordcdn.com", 
            "nordvpn.com", "spotify.com", "wikipedia.org", "yahoo.com", 
            "facebook.com", "instagram.com", "doubleclick.net", 
            "google-analytics.com", "googlesyndication.com", "senecacollege.ca"
        }
        
        # Load optional custom whitelist from whitelist.txt
        whitelist_path = os.path.join(SCRIPT_DIR, "..", "whitelist.txt")
        if os.path.exists(whitelist_path):
            try:
                with open(whitelist_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip().lower()
                        if line and not line.startswith("#"):
                            self.builtin_whitelist.add(line)
                print(f"[+] Loaded custom whitelist with {len(self.builtin_whitelist)} domains.")
            except Exception as e:
                print(f"[*] Warning: Failed to load whitelist.txt: {e}")
        
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

        # Start Telegram callback query long-polling listener
        if self.telegram_token != DEFAULT_TELEGRAM_TOKEN and self.telegram_chat_id != DEFAULT_TELEGRAM_CHAT_ID:
            threading.Thread(target=self._poll_telegram_updates, daemon=True).start()

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

    def extract_registered_domain(self, domain: str) -> str:
        """
        Extracts the core registered domain (SLD + TLD) from a full domain string.
        E.g. a1051.dscg4.akamai.net -> akamai.net
             repo42.cursor.sh -> cursor.sh
             my-sub.co.uk -> my-sub.co.uk
        """
        domain = domain.lower().strip()
        domain = domain.strip(".")
        
        parts = domain.split(".")
        if len(parts) <= 2:
            return domain
            
        # Common double TLDs where registered domain consists of 3 parts
        double_tlds = {
            "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", 
            "com.cn", "net.cn", "org.cn", "gov.cn", "com.tw", "org.tw", 
            "com.hk", "co.jp", "or.jp", "ne.jp", "ac.jp", "com.br", 
            "net.br", "com.au", "net.au", "org.au", "com.tr", "co.za", 
            "com.sg", "edu.sg"
        }
        
        last_two = ".".join(parts[-2:])
        if last_two in double_tlds:
            return ".".join(parts[-3:])
        else:
            return ".".join(parts[-2:])

    def trigger_defenses(self, domain: str, score: float):
        """Orchestrates Pi-hole DNS sinkhole, Suricata packet drop, and Telegram notification."""
        if self.auto_block:
            print(f"\n[!] TRIPPING NETWORK DEFENSES FOR MALICIOUS DOMAIN: {domain} (Score: {score:.4f})")
            
            # 1. Update Pi-hole custom list (local sinkhole DNS entry)
            self._block_on_pihole(domain)

            # 2. Add Suricata dynamic rule
            self._block_on_suricata(domain)
        else:
            print(f"\n[!] DETECTED SUSPICIOUS DOMAIN: {domain} (Score: {score:.4f}) - Alert-only mode (No Auto-Block)")

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
        if self.auto_block:
            message = f"🚨 *Malicious Indicator Detected!*\n*Domain:* `{domain}`\n*LSTM DGA Confidence Score:* `{score:.4f}`\n*Actions:* Blocked on Pi-hole & Suricata IDS (Auto-Block)"
            buttons = [
                [
                    {"text": "🚫 Confirm Block", "callback_data": f"block:{domain}"},
                    {"text": "🟢 Whitelist & Unblock", "callback_data": f"whitelist:{domain}"},
                    {"text": "⚪ Ignore & Unblock", "callback_data": f"ignore:{domain}"}
                ]
            ]
        else:
            message = f"🚨 *Suspicious Indicator Detected!*\n*Domain:* `{domain}`\n*LSTM DGA Confidence Score:* `{score:.4f}`\n*Actions:* Pending Administrator Decision..."
            buttons = [
                [
                    {"text": "🚫 Block Domain", "callback_data": f"block:{domain}"},
                    {"text": "🟢 Add to Whitelist", "callback_data": f"whitelist:{domain}"},
                    {"text": "⚪ Ignore Alert", "callback_data": f"ignore:{domain}"}
                ]
            ]

        print(f"[Alert] Sending Telegram notification...")

        # If placeholders are present, print and exit gracefully to facilitate testing
        if self.telegram_token == DEFAULT_TELEGRAM_TOKEN or self.telegram_chat_id == DEFAULT_TELEGRAM_CHAT_ID:
            print(f"[Mock Notification] Token/Chat ID unconfigured. Alert Details:\n{message}")
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "reply_markup": {
                "inline_keyboard": buttons
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                print("[+] Telegram threat alert dispatched successfully.")
            else:
                print(f"[-] Telegram dispatch failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[-] Telegram network connection failed: {e}")

    def _unblock_domain(self, domain: str):
        """Removes domain from Pi-hole custom list and Suricata rules."""
        print(f"[*] Reversing block for domain: {domain}...")
        
        # 1. Remove from Pi-hole
        try:
            if os.path.exists(self.pihole_custom_list):
                with open(self.pihole_custom_list, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                new_lines = [line for line in lines if domain not in line]
                with open(self.pihole_custom_list, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                print(f"[+] Removed '{domain}' from Pi-hole custom list.")
                
                # Reload DNS
                if self.mock:
                    print("[Mock] Reloading Pi-hole DNS: 'pihole restartdns reload'")
                else:
                    subprocess.run(["pihole", "restartdns", "reload"], check=True, capture_output=True)
        except Exception as e:
            print(f"[-] Failed to remove '{domain}' from Pi-hole: {e}")

        # 2. Remove from Suricata
        try:
            if os.path.exists(self.suricata_rules):
                with open(self.suricata_rules, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                new_lines = [line for line in lines if f'content:"{domain}"' not in line]
                with open(self.suricata_rules, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                print(f"[+] Removed '{domain}' from Suricata rules.")
                
                # Reload Suricata
                self.suricata_connector.reload_ruleset()
        except Exception as e:
            print(f"[-] Failed to remove '{domain}' from Suricata: {e}")

    def _whitelist_domain(self, domain: str):
        """Adds domain to the whitelist file and in-memory whitelist."""
        domain = domain.lower().strip()
        self.builtin_whitelist.add(domain)
        
        whitelist_path = os.path.join(SCRIPT_DIR, "..", "whitelist.txt")
        try:
            # Check if already in file
            already_in_file = False
            if os.path.exists(whitelist_path):
                with open(whitelist_path, "r", encoding="utf-8") as f:
                    if domain in f.read().lower():
                        already_in_file = True
            
            if not already_in_file:
                with open(whitelist_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{domain}\n")
                print(f"[+] Appended '{domain}' to whitelist.txt")
        except Exception as e:
            print(f"[-] Failed to write to whitelist.txt: {e}")

    def _poll_telegram_updates(self):
        """Long-polls Telegram getUpdates for callback query clicks."""
        print("[*] Starting Telegram interaction listener thread...")
        offset = 0
        
        while True:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
                params = {"offset": offset, "timeout": 30}
                response = requests.get(url, params=params, timeout=35)
                
                if response.status_code != 200:
                    time.sleep(5)
                    continue
                    
                data = response.json()
                if not data.get("ok"):
                    time.sleep(5)
                    continue
                    
                for update in data.get("result", []):
                    # Update offset to acknowledge processed items
                    offset = update["update_id"] + 1
                    
                    callback_query = update.get("callback_query")
                    if callback_query:
                        self._handle_callback_query(callback_query)
                        
            except Exception as e:
                print(f"[-] Error in Telegram interaction listener: {e}")
                time.sleep(5)

    def _handle_callback_query(self, callback_query):
        """Processes Telegram button clicks."""
        query_id = callback_query["id"]
        chat_id = callback_query["message"]["chat"]["id"]
        message_id = callback_query["message"]["message_id"]
        original_text = callback_query["message"].get("text", "")
        data = callback_query.get("data", "")
        
        if not data:
            return
            
        action, domain = data.split(":", 1)
        
        if action == "block":
            # Apply Pi-hole and Suricata blocks
            self._block_on_pihole(domain)
            self._block_on_suricata(domain)
            
            # Answer callback
            self._answer_callback(query_id, f"✅ Domain {domain} blocked successfully!")
            
            # Edit original message to remove buttons and show confirmation
            updated_text = (
                f"🚨 *Malicious Indicator Detected!*\n"
                f"*Domain:* `{domain}`\n"
                f"🔒 *Status:* Blocked by administrator."
            )
            self._edit_message(chat_id, message_id, updated_text)
            
        elif action == "whitelist":
            # 1. Reverse the block on Pi-hole & Suricata
            self._unblock_domain(domain)
            # 2. Add domain to whitelist
            self._whitelist_domain(domain)
            
            # Answer callback
            self._answer_callback(query_id, f"✅ Domain {domain} unblocked and whitelisted!")
            
            # Edit original message to remove buttons and show confirmation
            updated_text = (
                f"🚨 *Malicious Indicator Detected! (REVERSED)*\n"
                f"*Domain:* `{domain}`\n"
                f"✅ *Status:* Unblocked & Whitelisted by administrator."
            )
            self._edit_message(chat_id, message_id, updated_text)
            
        elif action == "ignore":
            # Reverse block if it was auto-blocked, otherwise just dismiss alert
            self._unblock_domain(domain)
            
            # Answer callback
            self._answer_callback(query_id, "Alert ignored.")
            
            # Edit original message to remove buttons
            updated_text = (
                f"🚨 *Malicious Indicator Detected!*\n"
                f"*Domain:* `{domain}`\n"
                f"⚪ *Status:* Alert ignored by administrator."
            )
            self._edit_message(chat_id, message_id, updated_text)

    def _answer_callback(self, query_id: str, text: str):
        """Answers a callback query to dismiss loading indicators in Telegram."""
        url = f"https://api.telegram.org/bot{self.telegram_token}/answerCallbackQuery"
        payload = {"callback_query_id": query_id, "text": text}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"[-] Failed to answer Telegram callback query: {e}")

    def _edit_message(self, chat_id: int, message_id: int, text: str):
        """Edits a Telegram message to remove inline keyboards and update contents."""
        url = f"https://api.telegram.org/bot{self.telegram_token}/editMessageText"
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": {"inline_keyboard": []} # Remove all buttons
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"[-] Failed to edit Telegram message: {e}")

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
                    clean_domain = domain.lower().strip()
                    
                    # 1. Skip IP addresses (reverse lookup addresses or absolute IPs)
                    if clean_domain.replace(".", "").isdigit():
                        continue
                        
                    # 2. Skip infrastructure, local, and reverse IP pointer domains
                    is_infra = False
                    infra_suffixes = [".arpa", ".local", ".lan", ".home", ".internal", ".onload", ".invalid", ".localhost", ".test", ".onion"]
                    for suffix in infra_suffixes:
                        if clean_domain.endswith(suffix):
                            is_infra = True
                            break
                    if is_infra:
                        continue
                        
                    # 3. Extract the registered domain for inference
                    reg_domain = self.extract_registered_domain(clean_domain)
                    
                    # 4. Check against whitelists (built-in and custom)
                    if reg_domain in self.builtin_whitelist:
                        print(f"[*] Whitelisted: {clean_domain} (Registered: {reg_domain}) - Skipping analysis.")
                        continue
                        
                    score = self.run_inference_on_domain(reg_domain)
                    print(f"[*] Scored: {clean_domain} (Registered: {reg_domain}) -> {score:.4f}")
                    
                    if score >= self.threshold:
                        self.trigger_defenses(clean_domain, score)


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
    parser.add_argument("--auto-block", action="store_true", default=None, help="Automatically block malicious domains upon detection (default: True)")
    parser.add_argument("--no-auto-block", action="store_false", dest="auto_block", help="Disable automatic blocking (manual mode via Telegram buttons)")
    
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

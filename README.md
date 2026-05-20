# LSTM-based DGA & Malicious Domain Detection Engine

A high-performance, real-time asynchronous network security agent. It parses incoming DNS queries from Pi-hole logs, evaluates domain structures using a Bidirectional LSTM neural network, and updates network defenses (Pi-hole local DNS blocks and Suricata drop rules) dynamically with ultra-low latency.

## Architecture Flow

```
                      [ Network Client ]
                              │  (DNS Query)
                              ▼
                        [ Pi-hole ] ──(Syncs Blocklist)──┐
                              │                           │
                      (Logs to syslog/file)               │
                              │                           │
                              ▼                           ▼
                     [ Python Log Agent ] ───► [ Trained LSTM Model ]
                              │                   (Inference Engine)
                        (If Malicious)                    │
                              │                           │
                              ▼                           ▼
                   [ Suricata / Telegram ] ◄──────────────┘
                    (IDS Alert & Drop Rule)
```

---

## Project Structure

```
dga-detection-engine/
├── data/                    # Dataset directory
├── models/                  # Stored model weights and vocabulary mapping
│   ├── dga_lstm_model.keras # Trained deep learning model
│   └── char_index.json      # Tokenizer character vocabulary
├── src/                     # Source code files
│   ├── __init__.py
│   ├── train.py             # LSTM training and evaluation pipeline
│   ├── agent.py             # Core real-time log-monitoring and inference daemon
│   └── suricata_socket.py   # High-speed Unix domain socket rule-reloader (JSON-RPC)
├── tests/                   # Python test suite
│   ├── __init__.py
│   ├── test_train.py        # Tokenizer, vocabulary, and generator tests
│   └── test_agent.py        # Log parser, preprocessor, and SID discovery tests
├── requirements.txt         # Package dependencies
├── dga-detector.service     # Systemd system service unit template
└── README.md                # System documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.9 - 3.11**
- For local testing, any platform (Windows, macOS, Linux) works.
- For production deployment, a Linux server/gateway running **Pi-hole** and **Suricata** is required.

### 2. Initialize Virtual Environment
Navigate to the project root and create a virtual environment:
```bash
python -m venv .venv
```

Activate the virtual environment:
* **Windows**:
  ```powershell
  .venv\Scripts\activate
  ```
* **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧠 Training the LSTM Model

The training script synthetically generates a balanced dataset of $8,000$ domain queries (half benign and half DGA across multiple generation families like high-entropy strings, vowel/consonant collisions, and dictionary permutations) to construct and compile a Bidirectional LSTM classifier.

To train the model:
```bash
python src/train.py
```

This will:
1. Map and store valid URL character tokens in `models/char_index.json`.
2. Generate synthetic domain sequences, perform training/testing split, and construct a Bidirectional LSTM network.
3. Output epochs progress, compile final performance indices (Accuracy, Precision, Recall, F1-Score, and ROC-AUC), and save the trained sequence model to `models/dga_lstm_model.keras`.

---

## 🧪 Running the Test Suite

A comprehensive test suite is provided to validate preprocessing shapes, log tailing, regex-based log parsing patterns, and SID sequence discovery calculations.

To run the automated tests:
```bash
pytest tests/
```

---

## 🚀 Running the Real-Time Agent

The agent is designed to run in two distinct modes:

### A. Development / Mock Mode (Cross-Platform)
Runs on Windows, macOS, or Linux. It creates a mock file system inside a `mock_env/` folder, starts a background simulation thread that writes random benign and malicious DNS query traffic, and prints live inference evaluations, blocklist additions, socket notifications, and mock Telegram logs in real-time.

```bash
python src/agent.py --mock
```

*Watch the terminal to see the background simulated traffic feed into the classifier, triggering block events on DGA detections!*

### B. Production Mode (Linux Gateway)
To run the agent on your active network gateway with real defenses enabled:

```bash
sudo .venv/bin/python src/agent.py \
  --threshold 0.85 \
  --telegram-token "YOUR_BOT_TOKEN" \
  --telegram-chat_id "YOUR_CHAT_ID"
```

*Note: Ensure you are running as `root` or using `sudo` so the agent has write permissions for syslog endpoints, local rule sheets, and reloading Pi-hole DNS.*

---

## ⚙️ Enterprise Integration Specifications

### 1. Direct Suricata Ruleset Socket Reload
Spawning a shell command such as `subprocess.run(["suricatasc", ...])` on every threat detection adds latency and CPU context-switching overhead. To achieve sub-millisecond defensive updates, the agent uses a custom `SuricataSocketConnector` (`src/suricata_socket.py`) which:
1. Establishes a direct connection to `/var/run/suricata/suricata-command.socket`.
2. Sends the command: `{"version": "0.2.0", "command": "ruleset-reload-nonblocking"}`.
3. Automatically falls back to standard `suricatasc` command shell execution if direct socket connection fails or is denied.

### 2. Auto-SID Incrementor
To prevent Suricata from rejecting rule additions due to duplicate Signature IDs (SIDs), the agent automatically scans your `/etc/suricata/rules/local.rules` file on startup, locates the highest SID present, and increments new signatures sequentially from that base (defaulting to starting from `1000001` if rules are empty).

### 3. Pi-hole Integration
When an indicator is identified, it is safely appended as `0.0.0.0 <domain>` to `/etc/pihole/custom.list` (avoiding duplicate writes). A non-blocking DNS reload command `pihole restartdns reload` is executed, instantly redirecting client lookups to a null route.

---

## 🛡️ Daemon Deployment (systemd)

To make the agent a permanent, self-healing background system service on your Linux gateway:

1. Copy the project folder to `/opt/dga-detection-engine`.
2. Copy the service unit template file to systemd:
   ```bash
   sudo cp dga-detector.service /etc/systemd/system/dga-detector.service
   ```
3. Reload systemd and enable/start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable dga-detector.service
   sudo systemctl start dga-detector.service
   ```
4. Verify system log outputs:
   ```bash
   sudo journalctl -u dga-detector.service -f
   ```

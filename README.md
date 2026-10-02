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
├── data/                         # Balanced training lists and provenance
│   ├── benign_domains.txt        # Majestic Million hostnames (label 0)
│   ├── malicious_domains.txt     # hagezi entropy DGA hostnames (label 1)
│   ├── provenance.json           # Feed versions, counts, checksums
│   ├── NOTICE                    # Source licenses and attribution
│   └── raw/                      # Original downloads (not committed; regenerate)
├── models/                       # Stored model weights and vocabulary mapping
│   ├── dga_lstm_model.keras      # Trained deep learning model
│   ├── dga_lstm_model.tflite     # Edge inference model
│   ├── char_index.json           # Tokenizer character vocabulary
│   └── training_metrics.json     # Accuracy, precision, recall, F1
├── src/                          # Source code files
│   ├── __init__.py
│   ├── download_datasets.py      # Fetches and balances the training lists
│   ├── domains.py                # Normalization and dataset assembly
│   ├── train.py                  # LSTM training and evaluation pipeline
│   ├── score_domains.py          # Offline confidence scores for domain names
│   ├── agent.py                  # Core real-time log-monitoring and inference daemon
│   └── suricata_socket.py        # High-speed Unix domain socket rule-reloader (JSON-RPC)
├── tests/                        # Python test suite
│   ├── __init__.py
│   ├── test_domains.py           # Normalizer and balancing tests
│   ├── test_train.py             # Tokenizer, vocabulary, and generator tests
│   ├── test_agent.py             # Log parser, preprocessor, and SID discovery tests
│   └── test_score_domains.py     # Offline scorer tokenization and CLI output
├── requirements.txt              # Package dependencies
├── dga-detector.service          # Systemd system service unit template
└── README.md                     # System documentation
```

---

## Installation & Setup

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

## Training the LSTM Model

Training uses real hostnames, with the two classes kept at equal size. Malicious names come from the [hagezi/nrd](https://github.com/hagezi/nrd) entropy DGA lists. Benign names come from the [Majestic Million](https://majestic.com/reports/majestic-million). The download script lowercases each name, strips schemes, ports, and paths, drops comments, rejects addresses that are not domain names, and de-duplicates.

### Download the lists

```bash
python src/download_datasets.py
```

Optional cap (the default `0` keeps every name that still balances):

```bash
python src/download_datasets.py --max-per-class 100000
```

That writes:

- `data/malicious_domains.txt` and `data/benign_domains.txt`
- `data/provenance.json` (versions, counts, checksums)
- `data/NOTICE` (licenses)
- `data/raw/` (original feeds; gitignored)

### Train

```bash
python src/train.py
```

This will:

1. Map and store valid URL character tokens in `models/char_index.json`.
2. Load the downloaded lists, pad or truncate each name to 45 characters, and split 80/20 with a stratified sample.
3. Train the CNN-LSTM for 5 epochs, print accuracy plus precision, recall, and F1 for the malicious class, and save `models/dga_lstm_model.keras`, `models/dga_lstm_model.tflite`, and `models/training_metrics.json`.

### Snapshot used for the committed model

Prepared 2026-10-01 from these feeds:

| Source | Window / list | Version | Normalized names |
| --- | --- | --- | --- |
| hagezi/nrd `dga7.txt` | past 7 days (primary) | 2026.1001.0630.05 | 588,072 |
| hagezi/nrd `dga14.txt` | past 14 days | 2026.1001.0630.39 | 1,241,888 |
| hagezi/nrd `dga30.txt` | past 30 days | 2026.1001.0631.48 | 2,531,076 |
| Majestic Million | daily CSV | Last-Modified 2026-10-01 05:00:19 GMT | 1,000,000 |

The 7-, 14-, and 30-day lists were all downloaded. Their union is 2,531,078 names. The 30-day list is almost a superset: only 2 names from the shorter lists are absent from it. The longer windows are what add older entropy registrations beyond the 7-day feed.

The training files are balanced at **997,679 benign** and **997,679 malicious**:

- Every 7-day name is kept (588,072).
- 409,607 further names are a deterministic sample (seed 42) of domains that appear on the 14- or 30-day lists and not on the 7-day list.
- 2,321 Majestic names that also appear on a hagezi list were removed from the benign class, which leaves 997,679 benign names. The malicious sample is cut to that same count so the classes stay equal.
- 2,145 selected names are longer than 45 characters and are truncated by the tokenizer.

Held-out evaluation uses a stratified 20% split (199,536 benign and 199,536 malicious). The positive class is malicious.

| Metric | Threshold 0.50 | Threshold 0.85 |
| --- | --- | --- |
| Accuracy | 0.9030 | 0.7699 |
| Precision (malicious) | 0.8582 | 0.9559 |
| Recall (malicious) | 0.9654 | 0.5660 |
| F1 (malicious) | 0.9087 | 0.7110 |
| ROC-AUC | 0.9608 | 0.9608 |

`python src/train.py` reports the 0.50 column. The agent blocks at 0.85 by default, which keeps malicious precision at 0.9559 and lowers recall. Full figures are in `models/training_metrics.json`.

The [1275.ru DGA feed](https://1275.ru/DGA/dga.txt) was not merged. It does not state a license, and this model is binary, so family labels are unused.

### Refresh the feed

hagezi marks the entropy lists as expiring after 8 hours. Majestic republishes its CSV daily. Re-download, then retrain, so overlap removal uses the same snapshot for both classes:

```bash
python src/download_datasets.py
python src/train.py
```

### Attribution

Malicious lists are from [hagezi/nrd](https://github.com/hagezi/nrd), licensed under the [GNU GPL-3.0](https://github.com/hagezi/nrd/blob/main/LICENSE). The files in `data/` are normalized subsets of those lists and stay under GPL-3.0. Copyright in the lists remains with their authors. See `data/NOTICE`.

Benign domains are from the Majestic Million ([CSV](https://downloads.majestic.com/majestic_million.csv), [description](https://majestic.com/reports/majestic-million)), licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Credit: Majestic. The CSV was reduced to hostnames, overlap with the hagezi lists was removed, and a deterministic sample was taken so the class counts match.

---

## Running the Test Suite

A comprehensive test suite is provided to validate preprocessing shapes, log tailing, regex-based log parsing patterns, and SID sequence discovery calculations.

To run the automated tests:
```bash
pytest tests/
```

Score names without starting the live agent. The script loads `models/dga_lstm_model.tflite` with `ai-edge-litert` or `tflite-runtime` when it can, and otherwise falls back to `models/dga_lstm_model.keras` (TensorFlow is imported only for that file). Each line is the domain, the confidence as a fraction and a percent, and `ALERT` or `ok` at threshold `0.85`:

```bash
python src/score_domains.py google.com qwertyuiopasdfghjkl.cc
```

Change the cutoff with `--threshold`. Names can also be piped on stdin, one per line.

---

## Running the Real-Time Agent

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

## Enterprise Integration Specifications

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

## Daemon Deployment (systemd)

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

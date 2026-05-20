import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from argparse import Namespace

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent import DGADetectorAgent

# Helper to create basic args namespace
def get_mock_args():
    return Namespace(
        mock=True,
        threshold=0.85,
        model="dummy_model.keras",
        vocab="dummy_vocab.json",
        log_path="dummy_pihole.log",
        custom_list="dummy_custom.list",
        rules_path="dummy_local.rules",
        socket_path="dummy.socket",
        telegram_token="YOUR_BOT_TOKEN",
        telegram_chat_id="YOUR_CHAT_ID"
    )

@patch("tensorflow.keras.models.load_model")
@patch("agent.open")
@patch("agent.json.load")
def test_agent_log_parsing(mock_json_load, mock_open, mock_load_model):
    """Verifies that the agent correctly parses DNS domain requests from standard syslog/pi-hole formats."""
    # Mock vocabulary
    mock_json_load.return_value = {"a": 1, "b": 2, "c": 3, ".": 4}
    
    agent = DGADetectorAgent(get_mock_args())
    
    # 1. Test standard A query
    log_line_1 = "May 20 01:25:26 dnsmasq[9876]: query[A] google.com from 192.168.1.100"
    domain_1 = agent.parse_dns_log_line(log_line_1)
    assert domain_1 == "google.com"

    # 2. Test standard AAAA query (IPv6)
    log_line_2 = "May 20 01:25:27 dnsmasq[9876]: query[AAAA] ipv6-dns.org from ::1"
    domain_2 = agent.parse_dns_log_line(log_line_2)
    assert domain_2 == "ipv6-dns.org"

    # 3. Test non-query lines (should return None)
    log_line_3 = "May 20 01:25:27 dnsmasq[9876]: reply google.com is 142.250.190.46"
    domain_3 = agent.parse_dns_log_line(log_line_3)
    assert domain_3 is None

    # 4. Test malformed query line (missing elements)
    log_line_4 = "May 20 01:25:27 dnsmasq[9876]: query[A] from 192.168.1.100"
    domain_4 = agent.parse_dns_log_line(log_line_4)
    assert domain_4 is None

@patch("tensorflow.keras.models.load_model")
@patch("agent.open")
@patch("agent.json.load")
def test_agent_preprocessing(mock_json_load, mock_open, mock_load_model):
    """Verifies that domain strings are correctly tokenized and padded by the agent."""
    mock_json_load.return_value = {"a": 1, "b": 2, "c": 3, "o": 4, "m": 5, ".": 6}
    
    agent = DGADetectorAgent(get_mock_args())
    
    # Tokenize "abc.com"
    processed = agent.preprocess_domain("abc.com")
    
    # Verify return shape (1, MAX_LEN)
    assert processed.shape == (1, 45)
    
    # "abc.com" has 7 characters: a=1, b=2, c=3, .=6, c=3, o=4, m=5
    # The rest should be padded with 0
    expected = [1, 2, 3, 6, 3, 4, 5] + [0] * (45 - 7)
    assert list(processed[0]) == expected

@patch("tensorflow.keras.models.load_model")
@patch("agent.open")
@patch("agent.json.load")
@patch("agent.os.path.exists")
def test_agent_sid_discovery(mock_exists, mock_json_load, mock_open, mock_load_model):
    """Verifies that the agent scans existing rules to find the next collision-free SID."""
    mock_json_load.return_value = {"a": 1}
    mock_exists.return_value = True
    
    # Mock opening the rules file and reading lines with mixed sids
    mock_file_content = [
        'drop dns any any -> any any (msg:"LSTM DGA Blocked: bad.com"; dns.query; content:"bad.com"; sid:1000005; rev:1;)\n',
        'drop dns any any -> any any (msg:"LSTM DGA Blocked: bad2.com"; dns.query; content:"bad2.com"; sid:1000012; rev:1;)\n',
        '# Some commented out rule with sid:1000050;\n',
        'drop dns any any -> any any (msg:"LSTM DGA Blocked: bad3.com"; dns.query; content:"bad3.com"; sid:1000020; rev:1;)\n'
    ]
    
    # Configure the file open mock
    mock_open_instance = mock_open.return_value.__enter__.return_value
    mock_open_instance.readlines.return_value = mock_file_content
    
    agent = DGADetectorAgent(get_mock_args())
    
    # The highest SID in mock content is 1000020 (ignoring commented rule 1000050 or checking if we count comments as rules too).
    # Since "sid:" is in all lines including comments, let's see how our code handles it.
    # Our code checks if "sid:" is in line, splits by "sid:", and extracts. So it parses both!
    # Expected starting SID = Max(1000005, 1000012, 1000050, 1000020) + 1 = 1000051.
    assert agent.current_sid == 1000051

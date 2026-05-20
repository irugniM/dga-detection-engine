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
        telegram_chat_id="YOUR_CHAT_ID",
        auto_block=None
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

@patch("tensorflow.keras.models.load_model")
@patch("agent.open")
@patch("agent.json.load")
def test_agent_whitelisting_and_extraction(mock_json_load, mock_open, mock_load_model):
    """Verifies registered domain extraction and whitelist skipping behaviors."""
    mock_json_load.return_value = {"a": 1}
    agent = DGADetectorAgent(get_mock_args())
    
    # 1. Verify standard core registered domain extraction
    assert agent.extract_registered_domain("a1051.dscg4.akamai.net") == "akamai.net"
    assert agent.extract_registered_domain("repo42.cursor.sh") == "cursor.sh"
    assert agent.extract_registered_domain("downloads77-windows.nordcdn.com") == "nordcdn.com"
    assert agent.extract_registered_domain("google.com") == "google.com"
    
    # 2. Verify double TLD extraction
    assert agent.extract_registered_domain("my-subdomain.co.uk") == "my-subdomain.co.uk"
    assert agent.extract_registered_domain("test.api.com.cn") == "api.com.cn"
    
    # 3. Verify built-in whitelist matching
    assert "akamai.net" in agent.builtin_whitelist
    assert "cursor.sh" in agent.builtin_whitelist
    assert "nordcdn.com" in agent.builtin_whitelist
    assert "google.com" in agent.builtin_whitelist
    assert "senecacollege.ca" in agent.builtin_whitelist


@patch("tensorflow.keras.models.load_model")
@patch("agent.open")
@patch("agent.json.load")
def test_agent_interactive_buttons_and_autoblock(mock_json_load, mock_open, mock_load_model):
    """Verifies that the agent respects auto_block settings and processes all interactive button callback query actions."""
    mock_json_load.return_value = {"a": 1}
    
    # 1. Verify auto-block options in constructor
    args_default = get_mock_args()
    args_default.auto_block = None
    with patch.dict(os.environ, {"AUTO_BLOCK": "false"}):
        agent_env_false = DGADetectorAgent(args_default)
        assert agent_env_false.auto_block is False

    with patch.dict(os.environ, {"AUTO_BLOCK": "true"}):
        agent_env_true = DGADetectorAgent(args_default)
        assert agent_env_true.auto_block is True

    # CLI override --no-auto-block (sets auto_block=False)
    args_no_auto = get_mock_args()
    args_no_auto.auto_block = False
    agent_cli_false = DGADetectorAgent(args_no_auto)
    assert agent_cli_false.auto_block is False

    # 2. Verify conditional blocking in trigger_defenses
    agent_cli_false._block_on_pihole = MagicMock()
    agent_cli_false._block_on_suricata = MagicMock()
    agent_cli_false._dispatch_alerts = MagicMock()

    # With auto_block = False
    agent_cli_false.trigger_defenses("malicious.biz", 0.95)
    agent_cli_false._block_on_pihole.assert_not_called()
    agent_cli_false._block_on_suricata.assert_not_called()
    agent_cli_false._dispatch_alerts.assert_called_once_with("malicious.biz", 0.95)

    # With auto_block = True
    agent_cli_false.auto_block = True
    agent_cli_false.trigger_defenses("malicious.biz", 0.95)
    agent_cli_false._block_on_pihole.assert_called_once_with("malicious.biz")
    agent_cli_false._block_on_suricata.assert_called_once_with("malicious.biz")

    # 3. Verify callback queries (_handle_callback_query)
    agent = DGADetectorAgent(get_mock_args())
    agent._block_on_pihole = MagicMock()
    agent._block_on_suricata = MagicMock()
    agent._unblock_domain = MagicMock()
    agent._whitelist_domain = MagicMock()
    agent._answer_callback = MagicMock()
    agent._edit_message = MagicMock()

    # A. Test action: "block"
    callback_block = {
        "id": "123",
        "message": {"chat": {"id": 999}, "message_id": 456, "text": "alert text"},
        "data": "block:badsite.info"
    }
    agent._handle_callback_query(callback_block)
    agent._block_on_pihole.assert_called_once_with("badsite.info")
    agent._block_on_suricata.assert_called_once_with("badsite.info")
    agent._answer_callback.assert_called_once_with("123", "✅ Domain badsite.info blocked successfully!")
    agent._edit_message.assert_called_once()

    # Reset mocks
    agent._block_on_pihole.reset_mock()
    agent._block_on_suricata.reset_mock()
    agent._answer_callback.reset_mock()
    agent._edit_message.reset_mock()

    # B. Test action: "whitelist"
    callback_whitelist = {
        "id": "124",
        "message": {"chat": {"id": 999}, "message_id": 456, "text": "alert text"},
        "data": "whitelist:badsite.info"
    }
    agent._handle_callback_query(callback_whitelist)
    agent._unblock_domain.assert_called_once_with("badsite.info")
    agent._whitelist_domain.assert_called_once_with("badsite.info")
    agent._answer_callback.assert_called_once_with("124", "✅ Domain badsite.info unblocked and whitelisted!")
    agent._edit_message.assert_called_once()

    # Reset mocks
    agent._unblock_domain.reset_mock()
    agent._whitelist_domain.reset_mock()
    agent._answer_callback.reset_mock()
    agent._edit_message.reset_mock()

    # C. Test action: "ignore"
    callback_ignore = {
        "id": "125",
        "message": {"chat": {"id": 999}, "message_id": 456, "text": "alert text"},
        "data": "ignore:badsite.info"
    }
    agent._handle_callback_query(callback_ignore)
    agent._unblock_domain.assert_called_once_with("badsite.info")
    agent._answer_callback.assert_called_once_with("125", "Alert ignored.")
    agent._edit_message.assert_called_once()


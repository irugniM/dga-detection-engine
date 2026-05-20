import sys
import os
import pytest
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from train import create_vocab, generate_synthetic_data, tokenize_and_pad, MAX_LEN

def test_vocab_creation():
    """Asserts that vocabulary is generated with proper characters and distinct positive IDs."""
    char_index = create_vocab()
    
    assert isinstance(char_index, dict)
    assert len(char_index) > 30
    assert "a" in char_index
    assert "0" in char_index
    assert "-" in char_index
    assert "." in char_index
    
    # 0 should be reserved for padding and not in vocabulary mapping directly
    assert 0 not in char_index.values()
    # Check that indices are unique positive integers
    assert len(set(char_index.values())) == len(char_index.values())

def test_generate_synthetic_data():
    """Asserts that data generator yields a balanced, structured dataset."""
    num_samples = 100
    domains, labels = generate_synthetic_data(num_samples=num_samples)
    
    assert len(domains) == num_samples
    assert len(labels) == num_samples
    # Balance check: 50% benign (0), 50% DGA (1)
    assert sum(labels) == num_samples // 2
    # Verify that all labels are binary
    assert set(labels) == {0, 1}

def test_tokenize_and_pad():
    """Asserts that tokenization shapes and pads domain sequences to the correct static tensor shapes."""
    char_index = create_vocab()
    test_domains = [
        "abc.com", # Short domain
        "a" * 100 + ".ru", # Long domain (needs truncation)
        "xyz-123.net" # Hyphenated domain
    ]
    
    X = tokenize_and_pad(test_domains, char_index)
    
    # Check shapes
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, MAX_LEN)
    
    # Check short domain post-padding (zeros at the end)
    # Length of "abc.com" is 7. Tokens should have non-zero for first 7, and 0s thereafter
    assert np.all(X[0, :7] > 0)
    assert np.all(X[0, 7:] == 0)
    
    # Check long domain truncation (exactly MAX_LEN and no zeros)
    assert np.all(X[1, :] > 0)
    assert len(X[1]) == MAX_LEN

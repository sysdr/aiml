# Day 170: Text Preprocessing and Tokenization

## Quick Start

```bash
# 1. Set up environment
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate

# 2. Run the lesson demo
python lesson_code.py

# 3. Run all tests
pytest test_lesson.py -v

# 4. Run with coverage report
pytest test_lesson.py -v --cov=lesson_code --cov-report=term-missing
```

## Expected Output (tests)

```
test_lesson.py::TestTextCleaning::test_lowercase                    PASSED
test_lesson.py::TestTextCleaning::test_html_tag_removal             PASSED
...
20 passed in ~8s
```

## What This Lesson Covers

| Concept | File Section |
|---|---|
| Text cleaning (HTML, unicode, whitespace) | `clean_text()` |
| Word-level tokenization | `word_tokenize()` |
| Vocabulary building with special tokens | `build_vocabulary()` |
| BERT WordPiece subword tokenization | `bert_tokenize()` |
| Full encoding with padding + attention mask | `bert_encode()` |
| Batch encoding | `bert_encode_batch()` |
| GPT-2 BPE tokenization | `gpt2_tokenize()` |
| Tokenizer comparison | `compare_tokenizers()` |
| Vocabulary statistics | `vocab_stats()` |
| Token length analysis | `token_length_analysis()` |

## Key Insight

Tokenizers are model components, not utilities.
A model trained with GPT-2's BPE tokenizer produces meaningless output if fed
BERT WordPiece token IDs, even for identical input text.
Vocabulary files are versioned alongside model weights in production.

## Files

```
day170_tokenization/
├── setup.sh          # Environment setup
├── requirements.txt  # Pinned dependencies
├── lesson_code.py    # Full preprocessing pipeline
├── test_lesson.py    # 20 pytest tests
└── README.md         # This file
```

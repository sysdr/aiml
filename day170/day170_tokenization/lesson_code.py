"""
Day 170: Text Preprocessing and Tokenization
============================================
Builds a complete NLP preprocessing pipeline from raw text to model-ready tensors.
Covers: cleaning, word tokenization, subword tokenization (BERT/GPT-2), encoding,
padding, attention masks, and batch processing.
"""

import re
import warnings
warnings.filterwarnings("ignore")

from transformers import BertTokenizer, GPT2Tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Text Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Stage 1 of the preprocessing pipeline.
    Removes HTML, normalizes unicode, collapses whitespace.
    """
    # Unescape common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")

    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize unicode apostrophes and quotes to ASCII
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')

    if lowercase:
        text = text.lower()

    # Collapse multiple whitespace characters
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_punctuation(text: str) -> str:
    """Remove non-alphanumeric characters (except whitespace)."""
    return re.sub(r'[^\w\s]', ' ', text)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: Word-Level Tokenization
# ──────────────────────────────────────────────────────────────────────────────

def word_tokenize(text: str) -> list:
    """
    Naive whitespace-based word tokenizer.
    Demonstrates the OOV problem and vocabulary explosion issue.
    """
    cleaned = remove_punctuation(clean_text(text))
    return [tok for tok in cleaned.split() if tok]


def build_vocabulary(corpus: list, max_vocab: int = 1000) -> dict:
    """
    Build a word-level vocabulary from a list of sentences.
    Returns: {word: id} mapping with special tokens prepended.
    """
    from collections import Counter

    all_tokens = []
    for sentence in corpus:
        all_tokens.extend(word_tokenize(sentence))

    freq = Counter(all_tokens)
    # Reserve IDs for special tokens
    special_tokens = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab = dict(special_tokens)

    for word, _ in freq.most_common(max_vocab - len(special_tokens)):
        vocab[word] = len(vocab)

    return vocab


def encode_word_level(text: str, vocab: dict) -> list:
    """Encode a sentence to integer IDs using a word-level vocabulary."""
    tokens = word_tokenize(text)
    unk_id = vocab.get("[UNK]", 1)
    return [vocab.get(tok, unk_id) for tok in tokens]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Subword Tokenization — BERT (WordPiece)
# ──────────────────────────────────────────────────────────────────────────────

def load_bert_tokenizer(model_name: str = "bert-base-uncased") -> BertTokenizer:
    """Load pretrained BERT WordPiece tokenizer."""
    return BertTokenizer.from_pretrained(model_name)


def bert_tokenize(tokenizer: BertTokenizer, text: str) -> list:
    """Return subword tokens (with ## continuation markers)."""
    return tokenizer.tokenize(text)


def bert_encode(tokenizer: BertTokenizer, text: str,
                max_length: int = 32) -> dict:
    """
    Full BERT encoding: tokens → IDs → padded tensor with attention mask.
    Returns a dict with 'input_ids' and 'attention_mask'.
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoding


def bert_encode_batch(tokenizer: BertTokenizer,
                      sentences: list,
                      max_length: int = 32) -> dict:
    """Encode a batch of sentences. All sequences padded to same length."""
    return tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Subword Tokenization — GPT-2 (BPE)
# ──────────────────────────────────────────────────────────────────────────────

def load_gpt2_tokenizer(model_name: str = "gpt2") -> GPT2Tokenizer:
    """Load pretrained GPT-2 BPE tokenizer."""
    tok = GPT2Tokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token  # GPT-2 has no PAD by default
    return tok


def gpt2_tokenize(tokenizer: GPT2Tokenizer, text: str) -> list:
    """Return BPE tokens (space prefix convention instead of ## suffix)."""
    return tokenizer.tokenize(text)


def compare_tokenizers(bert_tok: BertTokenizer,
                        gpt2_tok: GPT2Tokenizer,
                        words: list) -> dict:
    """
    Side-by-side comparison of BERT WordPiece vs GPT-2 BPE splits.
    Demonstrates that tokenizers are NOT interchangeable.
    """
    results = {}
    for word in words:
        results[word] = {
            "bert_wordpiece": bert_tok.tokenize(word),
            "gpt2_bpe":       gpt2_tok.tokenize(word),
        }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: Vocabulary Statistics
# ──────────────────────────────────────────────────────────────────────────────

def vocab_stats(tokenizer) -> dict:
    """Return key statistics about a tokenizer's vocabulary."""
    vocab_size = tokenizer.vocab_size
    special_tokens = tokenizer.all_special_tokens
    return {
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "num_special_tokens": len(special_tokens),
    }


def token_length_analysis(tokenizer, sentences: list) -> dict:
    """
    Analyze token count distribution across a corpus.
    Useful for choosing max_length when batching.
    """
    lengths = []
    for s in sentences:
        tokens = tokenizer.tokenize(s)
        lengths.append(len(tokens))

    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
        "lengths": lengths,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Day 170 — Text Preprocessing and Tokenization")
    print("=" * 60)

    # 1. Cleaning
    print("\n[1] Text Cleaning")
    raw = "The model's <em>accuracy</em> dropped by 12%!! &amp; nobody noticed."
    print(f"  Raw:     {raw}")
    print(f"  Cleaned: {clean_text(raw)}")

    # 2. Word tokenization
    print("\n[2] Word-Level Tokenization")
    corpus = [
        "Transformers revolutionized natural language processing.",
        "Tokenization splits text into meaningful units.",
        "Embeddings represent tokens as dense vectors.",
    ]
    vocab = build_vocabulary(corpus, max_vocab=50)
    print(f"  Vocabulary size: {len(vocab)}")
    encoded = encode_word_level(corpus[0], vocab)
    print(f"  Encoded: {encoded}")

    # 3. BERT tokenization
    print("\n[3] BERT WordPiece Tokenization")
    bert_tok = load_bert_tokenizer()

    test_sentence = "Tokenization splits rare words into subword pieces."
    tokens = bert_tokenize(bert_tok, test_sentence)
    print(f"  Tokens: {tokens}")

    encoding = bert_encode(bert_tok, test_sentence, max_length=16)
    print(f"  input_ids:      {encoding['input_ids'].tolist()}")
    print(f"  attention_mask: {encoding['attention_mask'].tolist()}")

    # 4. Batch encoding
    print("\n[4] Batch Encoding")
    batch = bert_encode_batch(bert_tok, corpus, max_length=24)
    print(f"  Batch shape: {list(batch['input_ids'].shape)}")
    print(f"  (3 sentences × 24 padded tokens)")

    # 5. GPT-2 BPE comparison
    print("\n[5] BERT vs GPT-2 Tokenizer Comparison")
    gpt2_tok = load_gpt2_tokenizer()
    comparison_words = ["unhappiness", "tokenization", "transformer", "GPT4"]
    comparison = compare_tokenizers(bert_tok, gpt2_tok, comparison_words)
    for word, splits in comparison.items():
        print(f"\n  '{word}'")
        print(f"    BERT : {splits['bert_wordpiece']}")
        print(f"    GPT-2: {splits['gpt2_bpe']}")

    # 6. Vocab statistics
    print("\n[6] Vocabulary Statistics")
    stats = vocab_stats(bert_tok)
    print(f"  BERT vocab size:    {stats['vocab_size']:,}")
    print(f"  Special tokens:     {stats['special_tokens']}")

    # 7. Token length analysis
    print("\n[7] Token Length Analysis")
    analysis = token_length_analysis(bert_tok, corpus)
    print(f"  Min: {analysis['min']}  Max: {analysis['max']}  Mean: {analysis['mean']}")
    print(f"  Per sentence: {analysis['lengths']}")

    print("\n" + "=" * 60)
    print("All sections complete. Run: pytest test_lesson.py -v")
    print("=" * 60)

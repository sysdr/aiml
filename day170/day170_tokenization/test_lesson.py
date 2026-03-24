"""
Test suite for Day 170: Text Preprocessing and Tokenization
============================================================
20 tests covering cleaning, word tokenization, vocabulary building,
BERT encoding, GPT-2 comparison, padding, attention masks, and batch processing.
"""

import pytest
import torch
from lesson_code import (
    clean_text,
    remove_punctuation,
    word_tokenize,
    build_vocabulary,
    encode_word_level,
    load_bert_tokenizer,
    bert_tokenize,
    bert_encode,
    bert_encode_batch,
    load_gpt2_tokenizer,
    gpt2_tokenize,
    compare_tokenizers,
    vocab_stats,
    token_length_analysis,
)


@pytest.fixture(scope="module")
def bert_tok():
    return load_bert_tokenizer()


@pytest.fixture(scope="module")
def gpt2_tok():
    return load_gpt2_tokenizer()


@pytest.fixture(scope="module")
def sample_corpus():
    return [
        "Transformers revolutionized natural language processing.",
        "Tokenization splits text into meaningful units.",
        "Embeddings represent tokens as dense vectors.",
    ]


# ── Section 1: Text Cleaning ──────────────────────────────────────────────────

class TestTextCleaning:

    def test_lowercase(self):
        assert clean_text("HELLO World") == "hello world"

    def test_html_tag_removal(self):
        result = clean_text("Hello <em>world</em>")
        assert "<em>" not in result
        assert "hello world" == result

    def test_html_entity_unescape(self):
        result = clean_text("cats &amp; dogs")
        assert "&amp;" not in result
        assert "&" in result

    def test_unicode_apostrophe_normalization(self):
        result = clean_text("don\u2019t")
        assert "\u2019" not in result
        assert "don't" in result

    def test_whitespace_collapse(self):
        result = clean_text("hello   world\t!")
        assert "  " not in result

    def test_remove_punctuation(self):
        result = remove_punctuation("hello, world!")
        assert "," not in result
        assert "!" not in result


# ── Section 2: Word Tokenization & Vocabulary ─────────────────────────────────

class TestWordTokenization:

    def test_basic_split(self):
        tokens = word_tokenize("the cat sat")
        assert tokens == ["the", "cat", "sat"]

    def test_punctuation_removed(self):
        tokens = word_tokenize("Hello, world!")
        assert "," not in tokens
        assert "!" not in tokens

    def test_empty_string(self):
        assert word_tokenize("") == []

    def test_vocabulary_contains_special_tokens(self, sample_corpus):
        vocab = build_vocabulary(sample_corpus)
        for tok in ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]:
            assert tok in vocab

    def test_vocabulary_special_token_ids(self, sample_corpus):
        vocab = build_vocabulary(sample_corpus)
        assert vocab["[PAD]"] == 0
        assert vocab["[UNK]"] == 1

    def test_encode_returns_integers(self, sample_corpus):
        vocab = build_vocabulary(sample_corpus)
        ids = encode_word_level(sample_corpus[0], vocab)
        assert all(isinstance(i, int) for i in ids)

    def test_oov_maps_to_unk(self, sample_corpus):
        vocab = build_vocabulary(sample_corpus, max_vocab=10)
        unk_id = vocab["[UNK]"]
        ids = encode_word_level("xyzzy frobnicator", vocab)
        assert all(i == unk_id for i in ids)


# ── Section 3: BERT WordPiece Tokenization ────────────────────────────────────

class TestBERTTokenization:

    def test_tokenize_returns_list(self, bert_tok):
        tokens = bert_tokenize(bert_tok, "hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_subword_continuation_marker(self, bert_tok):
        # "tokenization" should split into pieces, some with ##
        tokens = bert_tokenize(bert_tok, "tokenization")
        assert any(t.startswith("##") for t in tokens)

    def test_encode_returns_dict_with_keys(self, bert_tok):
        enc = bert_encode(bert_tok, "hello world", max_length=16)
        assert "input_ids" in enc
        assert "attention_mask" in enc

    def test_encode_respects_max_length(self, bert_tok):
        enc = bert_encode(bert_tok, "hello world", max_length=16)
        assert enc["input_ids"].shape[1] == 16

    def test_cls_token_at_position_0(self, bert_tok):
        enc = bert_encode(bert_tok, "hello world", max_length=16)
        cls_id = bert_tok.cls_token_id
        assert enc["input_ids"][0][0].item() == cls_id

    def test_attention_mask_shape_matches_input_ids(self, bert_tok):
        enc = bert_encode(bert_tok, "hello world", max_length=16)
        assert enc["attention_mask"].shape == enc["input_ids"].shape

    def test_padding_positions_have_zero_mask(self, bert_tok):
        # Short sentence padded to 32 → tail of mask should be zeros
        enc = bert_encode(bert_tok, "hi", max_length=32)
        mask = enc["attention_mask"][0].tolist()
        assert 0 in mask  # padding was applied


# ── Section 4: Batch Encoding ─────────────────────────────────────────────────

class TestBatchEncoding:

    def test_batch_shape_batch_size(self, bert_tok, sample_corpus):
        batch = bert_encode_batch(bert_tok, sample_corpus, max_length=32)
        assert batch["input_ids"].shape[0] == len(sample_corpus)

    def test_batch_shape_sequence_dim(self, bert_tok, sample_corpus):
        batch = bert_encode_batch(bert_tok, sample_corpus, max_length=32)
        assert batch["input_ids"].shape[1] == 32


# ── Section 5: Tokenizer Comparison ──────────────────────────────────────────

class TestTokenizerComparison:

    def test_bert_and_gpt2_differ_on_same_word(self, bert_tok, gpt2_tok):
        word = "unhappiness"
        bert_tokens = bert_tok.tokenize(word)
        gpt2_tokens = gpt2_tok.tokenize(word)
        # They use different merge vocabularies → different splits
        assert bert_tokens != gpt2_tokens

    def test_compare_returns_all_words(self, bert_tok, gpt2_tok):
        words = ["tokenization", "transformer"]
        result = compare_tokenizers(bert_tok, gpt2_tok, words)
        assert set(result.keys()) == set(words)


# ── Section 6: Vocab Stats & Length Analysis ─────────────────────────────────

class TestVocabAndLengthAnalysis:

    def test_bert_vocab_size_range(self, bert_tok):
        stats = vocab_stats(bert_tok)
        # bert-base-uncased has ~30,522 tokens
        assert 28_000 < stats["vocab_size"] < 35_000

    def test_token_length_analysis_keys(self, bert_tok, sample_corpus):
        result = token_length_analysis(bert_tok, sample_corpus)
        for key in ["min", "max", "mean", "lengths"]:
            assert key in result

    def test_token_length_analysis_count(self, bert_tok, sample_corpus):
        result = token_length_analysis(bert_tok, sample_corpus)
        assert len(result["lengths"]) == len(sample_corpus)

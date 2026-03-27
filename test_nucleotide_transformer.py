"""
Tests for NucleotideTransformerModel and DNA Featurizers
=========================================================
Mirrors DeepChem test conventions (test_chemberta.py, test_molformer.py).

Run:
    pytest deepchem/models/torch_models/tests/test_nucleotide_transformer.py -v
"""

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch        = pytest.importorskip("torch")
dc           = pytest.importorskip("deepchem")

from deepchem.models.torch_models.nucleotide_transformer import (
    NucleotideTransformerModel,
    NUCLEOTIDE_TRANSFORMER_MODELS,
)
from deepchem.feat.sequence_featurizers.dna_tokenizer_featurizer import (
    DNATokenizerFeaturizer,
    KMerDNAFeaturizer,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

SMALL_MODEL = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
MAX_LEN     = 64

DNA_SEQS = [
    "ATCGATCGATCGATCGATCG",
    "GCTAGCTAGCTAGCTAGCTA",
    "TTTTAAAACCCCGGGG",
    "ACGTACGTACGTACGT",
    "GGGGCCCCAAAATTTT",
    "ATATATATATATATAT",
]
LABELS_CLF = np.array([[1], [0], [1], [0], [1], [0]], dtype=np.float32)
LABELS_REG = np.array([[0.5], [1.2], [-0.3], [0.8], [2.1], [-1.0]], dtype=np.float32)


def make_clf_dataset():
    return dc.data.NumpyDataset(
        X=np.array(DNA_SEQS, dtype=object), y=LABELS_CLF)

def make_reg_dataset():
    return dc.data.NumpyDataset(
        X=np.array(DNA_SEQS, dtype=object), y=LABELS_REG)


# ── featurizer tests ──────────────────────────────────────────────────────────

class TestDNATokenizerFeaturizer:

    def test_output_shape(self):
        feat = DNATokenizerFeaturizer(SMALL_MODEL, max_length=MAX_LEN)
        out  = feat.featurize(DNA_SEQS)
        assert out.shape == (len(DNA_SEQS), MAX_LEN)

    def test_integer_tokens(self):
        feat = DNATokenizerFeaturizer(SMALL_MODEL, max_length=MAX_LEN)
        out  = feat.featurize(DNA_SEQS[:2])
        assert np.issubdtype(out.dtype, np.integer)

    def test_attention_mask_shape(self):
        feat = DNATokenizerFeaturizer(SMALL_MODEL, max_length=MAX_LEN,
                                      return_attention_mask=True)
        out  = feat.featurize(DNA_SEQS[:3])
        assert out.shape == (3, 2, MAX_LEN)

    def test_short_sequence_padded(self):
        feat = DNATokenizerFeaturizer(SMALL_MODEL, max_length=MAX_LEN)
        out  = feat.featurize(["AT"])
        assert out.shape == (1, MAX_LEN)

    def test_long_sequence_truncated(self):
        feat = DNATokenizerFeaturizer(SMALL_MODEL, max_length=MAX_LEN)
        out  = feat.featurize(["ATCG" * 200])   # >> MAX_LEN
        assert out.shape == (1, MAX_LEN)


class TestKMerDNAFeaturizer:

    def test_3mer_shape(self):
        feat = KMerDNAFeaturizer(k=3)
        out  = feat.featurize(DNA_SEQS)
        assert out.shape == (len(DNA_SEQS), 4**3)

    def test_6mer_shape(self):
        feat = KMerDNAFeaturizer(k=6)
        out  = feat.featurize(DNA_SEQS[:2])
        assert out.shape == (2, 4**6)

    def test_normalized_sums_to_one(self):
        feat = KMerDNAFeaturizer(k=3, normalize=True)
        out  = feat.featurize(DNA_SEQS[:2])
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_unnormalized_counts(self):
        feat = KMerDNAFeaturizer(k=3, normalize=False)
        out  = feat.featurize(["ATCATC"])   # ATCATC has 4 3-mers
        assert out.sum() == 4


# ── model tests ───────────────────────────────────────────────────────────────

class TestNucleotideTransformerModel:

    @pytest.fixture
    def clf_model(self, tmp_path):
        return NucleotideTransformerModel(
            n_tasks=1, mode="classification",
            model_path=SMALL_MODEL,
            max_seq_length=MAX_LEN, batch_size=2,
            model_dir=str(tmp_path))

    @pytest.fixture
    def reg_model(self, tmp_path):
        return NucleotideTransformerModel(
            n_tasks=1, mode="regression",
            model_path=SMALL_MODEL,
            max_seq_length=MAX_LEN, batch_size=2,
            model_dir=str(tmp_path))

    # ── smoke ──────────────────────────────────────────────────────────────

    def test_classification_fit_predict(self, clf_model):
        ds   = make_clf_dataset()
        loss = clf_model.fit(ds, nb_epoch=1)
        assert loss >= 0.0
        pred = clf_model.predict(ds)
        assert pred.shape == (len(DNA_SEQS), 1)

    def test_regression_fit_predict(self, reg_model):
        ds   = make_reg_dataset()
        loss = reg_model.fit(ds, nb_epoch=1)
        pred = reg_model.predict(ds)
        assert pred.shape == (len(DNA_SEQS), 1)

    def test_predictions_are_finite(self, clf_model):
        ds   = make_clf_dataset()
        pred = clf_model.predict(ds)
        assert np.all(np.isfinite(pred))

    # ── embeddings ─────────────────────────────────────────────────────────

    def test_embeddings_mean_shape(self, clf_model):
        embs   = clf_model.get_embeddings(DNA_SEQS, pooling="mean")
        hidden = clf_model.model.backbone.config.hidden_size
        assert embs.shape == (len(DNA_SEQS), hidden)

    def test_embeddings_cls_shape(self, clf_model):
        embs   = clf_model.get_embeddings(DNA_SEQS, pooling="cls")
        hidden = clf_model.model.backbone.config.hidden_size
        assert embs.shape == (len(DNA_SEQS), hidden)

    def test_embeddings_finite(self, clf_model):
        embs = clf_model.get_embeddings(DNA_SEQS)
        assert np.all(np.isfinite(embs))

    # ── save / restore ─────────────────────────────────────────────────────

    def test_save_restore(self, clf_model, tmp_path):
        ds     = make_clf_dataset()
        _      = clf_model.fit(ds, nb_epoch=1)
        before = clf_model.predict(ds)
        clf_model.save_checkpoint()

        restored = NucleotideTransformerModel(
            n_tasks=1, mode="classification",
            model_path=SMALL_MODEL,
            max_seq_length=MAX_LEN, batch_size=2,
            model_dir=clf_model.model_dir)
        restored.restore()
        after = restored.predict(ds)

        np.testing.assert_allclose(before, after, rtol=1e-4)

    # ── edge cases ─────────────────────────────────────────────────────────

    def test_long_sequence_truncated(self, clf_model):
        long = ["ATCG" * 400]
        ds   = dc.data.NumpyDataset(
            X=np.array(long, dtype=object),
            y=np.array([[1]], dtype=np.float32))
        pred = clf_model.predict(ds)
        assert pred.shape == (1, 1)

    def test_frozen_backbone(self, tmp_path):
        model = NucleotideTransformerModel(
            n_tasks=1, mode="classification",
            model_path=SMALL_MODEL,
            max_seq_length=MAX_LEN, batch_size=2,
            freeze_backbone=True,
            model_dir=str(tmp_path))
        ds   = make_clf_dataset()
        loss = model.fit(ds, nb_epoch=1)
        assert loss >= 0.0

    def test_pretrain_raises(self, clf_model):
        ds = make_clf_dataset()
        with pytest.raises(NotImplementedError):
            clf_model.pretrain(ds)

    # ── model registry ─────────────────────────────────────────────────────

    def test_model_registry_keys(self):
        assert "v2-100m-multi-species" in NUCLEOTIDE_TRANSFORMER_MODELS
        assert "2.5b-multi-species"    in NUCLEOTIDE_TRANSFORMER_MODELS
        for key, hf_id in NUCLEOTIDE_TRANSFORMER_MODELS.items():
            assert hf_id.startswith("InstaDeepAI/")

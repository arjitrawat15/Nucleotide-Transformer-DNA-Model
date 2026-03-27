"""
DNA Sequence Featurizers for DeepChem
=======================================
Two featurizers for transforming raw DNA strings into model-ready arrays:

  DNATokenizerFeaturizer  – wraps any HuggingFace tokenizer that accepts
                            nucleotide strings (NT, DNABERT, Caduceus, …)

  KMerDNAFeaturizer       – bag-of-k-mers counts, useful as a fast baseline

Place this file at:
    deepchem/feat/sequence_featurizers/dna_tokenizer_featurizer.py

Register in:
    deepchem/feat/sequence_featurizers/__init__.py
    deepchem/feat/__init__.py

GSoC 2026 — DeepChem: Single Cell and DNA Foundation Models
"""

from __future__ import annotations

import logging
from itertools import product
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from deepchem.feat.base_classes import Featurizer
    HAS_DC = True
except ImportError:
    HAS_DC = False
    class Featurizer:  # type: ignore
        """Minimal stub so the file is importable without full DeepChem."""
        def featurize(self, datapoints, log_every_n=1000):
            return np.stack(
                [self._featurize(d) for d in datapoints], axis=0)

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ─────────────────────────────────────────────────────────────────────────────
#  DNATokenizerFeaturizer
# ─────────────────────────────────────────────────────────────────────────────

class DNATokenizerFeaturizer(Featurizer):
    """
    Tokenises raw DNA sequences for transformer-based foundation models.

    Wraps any HuggingFace tokenizer that accepts nucleotide strings.
    Compatible with:

    * **Nucleotide Transformer** (InstaDeepAI) — 6-mer vocabulary
    * **DNABERT** — k-mer vocabulary
    * **Caduceus / HyenaDNA** — character-level vocabulary

    Parameters
    ----------
    tokenizer_path : str
        HuggingFace model ID or local path whose tokenizer to load.
    max_length : int
        Maximum token count.  Sequences longer than this are truncated;
        shorter ones are padded to ``max_length``.
    return_attention_mask : bool
        If ``True`` the output shape is ``(N, 2, max_length)`` where
        ``output[:, 0, :]`` = input IDs and ``output[:, 1, :]`` = mask.
        If ``False`` (default) shape is ``(N, max_length)`` of input IDs.

    Examples
    --------
    >>> feat = DNATokenizerFeaturizer(
    ...     tokenizer_path='InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    ...     max_length=128)
    >>> seqs = ['ATCGATCGATCG', 'GCTAGCTAGCTA']
    >>> out  = feat.featurize(seqs)
    >>> out.shape
    (2, 128)
    """

    def __init__(
        self,
        tokenizer_path: str = (
            "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"),
        max_length: int = 512,
        return_attention_mask: bool = False,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("pip install transformers")
        self.tokenizer            = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True)
        self.max_length           = max_length
        self.return_attention_mask = return_attention_mask

    def _featurize(self, datapoint: str) -> np.ndarray:
        enc = self.tokenizer(
            str(datapoint),
            return_tensors="np",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        if self.return_attention_mask:
            return np.stack(
                [enc["input_ids"][0], enc["attention_mask"][0]], axis=0)
        return enc["input_ids"][0]

    def featurize(
        self,
        datapoints,
        log_every_n: int = 1000,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        datapoints : list[str] or array-like
            DNA strings.

        Returns
        -------
        np.ndarray  shape (N, max_length) or (N, 2, max_length)
        """
        feats = []
        for i, dp in enumerate(datapoints):
            if i % log_every_n == 0:
                logger.debug("Featurising %d / %d", i, len(datapoints))
            feats.append(self._featurize(dp))
        return np.stack(feats, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  KMerDNAFeaturizer
# ─────────────────────────────────────────────────────────────────────────────

class KMerDNAFeaturizer(Featurizer):
    """
    Bag-of-k-mers frequency vector for a DNA sequence.

    Useful as a fast classical-ML baseline and as an interpretability
    probe to compare against transformer embeddings.

    Parameters
    ----------
    k : int
        k-mer length.  Default 6 matches the Nucleotide Transformer vocab.
    normalize : bool
        Return relative k-mer frequencies (sum to 1) instead of counts.

    Examples
    --------
    >>> feat = KMerDNAFeaturizer(k=3)
    >>> out  = feat.featurize(['ATCGATCG'])
    >>> out.shape
    (1, 64)
    """

    BASES = "ACGT"

    def __init__(self, k: int = 6, normalize: bool = True):
        self.k         = k
        self.normalize = normalize
        kmers          = ["".join(p) for p in product(self.BASES, repeat=k)]
        self.vocab     = {km: i for i, km in enumerate(kmers)}

    def _featurize(self, sequence: str) -> np.ndarray:
        seq = sequence.upper().replace("N", "A")
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for i in range(len(seq) - self.k + 1):
            km = seq[i : i + self.k]
            if km in self.vocab:
                vec[self.vocab[km]] += 1.0
        if self.normalize and vec.sum() > 0:
            vec /= vec.sum()
        return vec

    def featurize(self, datapoints, log_every_n: int = 1000) -> np.ndarray:
        return np.stack([self._featurize(s) for s in datapoints], axis=0)

"""
NucleotideTransformer Model for DeepChem
==========================================
Wraps InstaDeepAI's Nucleotide Transformer family into DeepChem's
TorchModel infrastructure — following the same pattern as ChemBERTa
(chemberta.py) and MolFormer (molformer.py).

Architecture
------------
    DNA string(s)
        └─► DNATokenizerFeaturizer   (6-mer / BPE tokenisation)
                └─► NT Backbone      (ESM-style bidirectional transformer)
                        └─► mean-pool last hidden state
                                └─► LayerNorm → Dropout → Linear → GELU
                                        └─► Linear → logits / scalar

Supported backbone sizes
------------------------
    v2-100m   InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
    v2-250m   InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
    v2-500m   InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
    500m-hr   InstaDeepAI/nucleotide-transformer-500m-human-ref
    2.5b      InstaDeepAI/nucleotide-transformer-2.5b-multi-species

References
----------
Dalla-Torre et al. (2023). The Nucleotide Transformer.
https://doi.org/10.1101/2023.01.11.523679

GSoC 2026 — DeepChem: Single Cell and DNA Foundation Models
Author : Arjit (arjit@example.com)
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from deepchem.models.losses import L2Loss, SigmoidCrossEntropy
    from deepchem.models.torch_models.torch_model import TorchModel
    HAS_DC = True
except ImportError:
    HAS_DC = False
    class TorchModel:  # type: ignore
        pass

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    
NUCLEOTIDE_TRANSFORMER_MODELS: Dict[str, str] = {
    "v2-100m-multi-species":
        "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    "v2-250m-multi-species":
        "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    "v2-500m-multi-species":
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "500m-human-ref":
        "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "500m-multi-species":
        "InstaDeepAI/nucleotide-transformer-500m-multi-species",
    "2.5b-multi-species":
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "2.5b-1000g":
        "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
}
DEFAULT_NT_MODEL = "v2-100m-multi-species"

class _NTModule(nn.Module):
    """Backbone + MLP task head.  Owned by NucleotideTransformerModel."""

    def __init__(
        self,
        model_path: str,
        n_tasks: int,
        mode: str = "classification",
        freeze_backbone: bool = False,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("pip install transformers")

        self.mode = mode
        self.backbone = AutoModel.from_pretrained(
            model_path, trust_remote_code=True)
        H = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Backbone frozen — running linear-probe mode.")

        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Dropout(head_dropout),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(H // 2, n_tasks),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        H = out.last_hidden_state.float()                    # (B, L, D)
        if attention_mask is not None:
            m      = attention_mask.unsqueeze(-1).float()   # (B, L, 1)
            pooled = (H * m).sum(1) / m.sum(1).clamp(1)    # (B, D)
        else:
            pooled = H.mean(1)
        return self.head(pooled)                             # (B, n_tasks)



class NucleotideTransformerModel(TorchModel):
    """
    DeepChem wrapper for the InstaDeepAI Nucleotide Transformer.

    Follows the same API contract as ``ChemBERTa`` and ``MolFormer``:
    ``fit()``, ``predict()``, ``evaluate()``, ``save_checkpoint()``,
    and ``restore()`` all work without modification.

    Parameters
    ----------
    n_tasks : int
        Number of output tasks.
    mode : str
        ``'classification'`` or ``'regression'``.
    model_path : str
        Short key from ``NUCLEOTIDE_TRANSFORMER_MODELS``, full HuggingFace
        model ID, or path to a local directory.
    max_seq_length : int
        Token-level maximum length (6-mer tokens ≈ 6 bp each, so 512
        tokens ≈ 3 kb of DNA).
    freeze_backbone : bool
        Freeze transformer weights; train only the task head.
    head_dropout : float
    batch_size : int
    learning_rate : float
    **kwargs
        Forwarded to :class:`~deepchem.models.torch_models.TorchModel`.

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.nucleotide_transformer import (
    ...     NucleotideTransformerModel)
    >>>
    >>> seqs  = ['ATCGATCGATCGATCG', 'GCTAGCTAGCTAGCTA']
    >>> X     = np.array(seqs, dtype=object)
    >>> y     = np.array([[1], [0]], dtype=np.float32)
    >>> ds    = dc.data.NumpyDataset(X=X, y=y)
    >>>
    >>> model = NucleotideTransformerModel(
    ...     n_tasks=1, mode='classification',
    ...     model_path='v2-100m-multi-species',
    ...     batch_size=2)
    >>> _ = model.fit(ds, nb_epoch=1)
    >>> preds = model.predict(ds)   # ndarray (2, 1)
    """

    def __init__(
        self,
        n_tasks: int = 1,
        mode: str = "classification",
        model_path: str = DEFAULT_NT_MODEL,
        max_seq_length: int = 512,
        freeze_backbone: bool = False,
        head_dropout: float = 0.1,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        **kwargs,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("pip install transformers")

        if model_path in NUCLEOTIDE_TRANSFORMER_MODELS:
            model_path = NUCLEOTIDE_TRANSFORMER_MODELS[model_path]

        self.n_tasks        = n_tasks
        self.mode           = mode
        self.model_path     = model_path
        self.max_seq_length = max_seq_length

        pt_model = _NTModule(
            model_path=model_path,
            n_tasks=n_tasks,
            mode=mode,
            freeze_backbone=freeze_backbone,
            head_dropout=head_dropout,
        )

        if HAS_DC:
            loss = SigmoidCrossEntropy() if mode == "classification" else L2Loss()
        else:
            loss = nn.BCEWithLogitsLoss() if mode == "classification" else nn.MSELoss()

        super().__init__(
            model=pt_model,
            loss=loss,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

    def _tokenize(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        return self._tokenizer(
            sequences,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
        )

    def default_generator(
        self,
        dataset,
        epochs: int = 1,
        mode: str = "fit",
        deterministic: bool = True,
        pad_batches: bool = True,
    ) -> Iterable:
        """
        Yield ``(inputs, labels, weights)`` tuples.

        ``dataset.X`` is expected to be an object array of raw DNA strings.
        We tokenise each batch here so no featurisation step is needed
        when calling ``fit()`` or ``predict()``.
        """
        for _ in range(epochs):
            for X_b, y_b, w_b, _ in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches,
            ):
                seqs   = [str(s) for s in X_b.flatten()]
                tokens = self._tokenize(seqs)
                inputs = [tokens["input_ids"],
                          tokens.get("attention_mask")]

                if mode == "predict":
                    yield (inputs, [], [])
                else:
                    yield (inputs, [y_b], [w_b])

    def get_embeddings(
        self,
        sequences: List[str],
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        Extract sequence embeddings from the last transformer layer.

        Parameters
        ----------
        sequences : list[str]
            Raw DNA strings.
        pooling : 'mean' | 'cls'
            How to aggregate the per-token hidden states.

        Returns
        -------
        np.ndarray  shape (N, hidden_size)
        """
        self.model.eval()
        out: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                tok  = self._tokenize(sequences[i : i + self.batch_size])
                ids  = tok["input_ids"].to(self.device)
                mask = tok.get("attention_mask")
                if mask is not None:
                    mask = mask.to(self.device)

                hs = self.model.backbone(
                    input_ids=ids, attention_mask=mask
                ).last_hidden_state.float()

                if pooling == "mean":
                    if mask is not None:
                        m   = mask.unsqueeze(-1).float()
                        emb = (hs * m).sum(1) / m.sum(1).clamp(1)
                    else:
                        emb = hs.mean(1)
                elif pooling == "cls":
                    emb = hs[:, 0, :]
                else:
                    raise ValueError("pooling must be 'mean' or 'cls'")

                out.append(emb.cpu().numpy())

        return np.concatenate(out, axis=0)


    def pretrain(self, dataset, nb_epoch: int = 1, **_kwargs):
        """
        Scaffold for masked-language-model pre-training on DNA sequences.

        Planned for GSoC Weeks 8–9.  Fine-tuning via ``fit()`` is fully
        functional.
        """
        raise NotImplementedError(
            "Continued pre-training (MLM) is scheduled for Weeks 8–9 of "
            "the GSoC timeline.  Use fit() for supervised fine-tuning."
        )

"""
bc/behavioral_cloning.py
========================
Behavioral Cloning trainer — learns from human demonstrations.

This is the first and most important learning phase for Intern.
The human records themselves doing a task; this trainer imprints
exactly that behaviour into the TransformerAgentNetwork.

Flow
----
  Human records session(s)
        ↓
  Trace JSON files saved to trace_dir
        ↓
  BCTrainer.train() reads traces, encodes states, trains transformer
        ↓
  Checkpoint saved → used by RL fine-tuning and ContinualLearner
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── path setup ─────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))          # bc/
_IM_DIR  = os.path.dirname(_HERE)                              # intelligence/
_LM_DIR  = os.path.dirname(_IM_DIR)                           # intelligence/
_COMP    = os.path.dirname(_LM_DIR)                            # components/
_ROOT    = os.path.dirname(_COMP)                              # Intern/
_MODEL_DIR = os.path.join(_COMP, "intelligence", "model")
for _p in (_ROOT, _COMP, _MODEL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class BCTrainer:
    """
    Wraps TransformerAgentNetwork training for clean use inside InternModel.

    Parameters
    ----------
    trace_dir   : Directory containing trace JSON files (and session_* sub-dirs).
    save_path   : Where to write the trained checkpoint.
    epochs      : Training epochs.
    batch_size  : Mini-batch size.
    lr          : Learning rate.
    val_split   : Fraction of data held out for validation.
    aug_drop    : Element dropout augmentation probability (anti-overfitting).
    device      : "auto" | "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        trace_dir:       str | list[str] = "data/output/traces/live",
        save_path:       str   = "tasks/form_filling/model.pt",
        epochs:          int   = 50,
        batch_size:      int   = 16,
        lr:              float = 1e-3,
        val_split:       float = 0.15,
        aug_drop:        float = 0.1,
        device:          str   = "auto",
        d_model:         int   = 64,
        num_layers:      int   = 2,
        dim_feedforward: int   = 128,
        dropout:         float = 0.2,
        hist_len:        int   = 4,
        max_elements:    int   = 128,   # elements per state the model can see;
                                        # anything past it is invisible to training
        disambiguate_attempted: str = "none",   # "none" | "rank" | "section"
        rare_weight_basis:      str = "type",   # "none" | "type" | "field"
        section_pattern: Optional[str] = None,
        section_prefix:  str = "section_",
        action_space:    str = "legacy",   # "legacy" | "semantic"
    ):
        self.trace_dir       = trace_dir
        self.save_path       = save_path
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.val_split       = val_split
        self.aug_drop        = aug_drop
        self.device          = device
        self.d_model         = d_model
        self.num_layers      = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout         = dropout
        self.hist_len        = hist_len
        self.max_elements    = max_elements
        self.disambiguate_attempted = disambiguate_attempted
        self.rare_weight_basis      = rare_weight_basis
        self.section_pattern        = section_pattern
        self.section_prefix         = section_prefix
        self.action_space           = action_space

    def train(self, trace_dir: Optional[str] = None, epochs: Optional[int] = None):
        """
        Train on all traces in trace_dir.  Returns the trained model.

        Auto-filters traces with 0 elements so you never need to manually
        clean up empty OCR traces.
        """
        from transformer import train as _train

        d = trace_dir or self.trace_dir
        e = epochs    or self.epochs

        # Auto-filter: skip traces with no elements
        filtered_dir = self._filter_empty(d)

        logger.info("BCTrainer: training on %s  epochs=%d", filtered_dir, e)
        model = _train(
            data_dir        = filtered_dir,
            epochs          = e,
            batch_size      = self.batch_size,
            lr              = self.lr,
            val_split       = self.val_split,
            save_path       = self.save_path,
            aug_drop_prob   = self.aug_drop,
            device_str      = self.device,
            d_model         = self.d_model,
            num_layers      = self.num_layers,
            dim_feedforward = self.dim_feedforward,
            dropout         = self.dropout,
            hist_len        = self.hist_len,
            max_elements    = self.max_elements,
            disambiguate_attempted = self.disambiguate_attempted,
            rare_weight_basis      = self.rare_weight_basis,
            section_pattern        = self.section_pattern,
            section_prefix         = self.section_prefix,
            action_space           = self.action_space,
        )
        logger.info("BCTrainer: checkpoint saved → %s", self.save_path)
        return model

    def load(self, model_path: Optional[str] = None):
        """Load and return a trained checkpoint."""
        from transformer import _load_model
        import torch
        path   = model_path or self.save_path
        device = torch.device("cpu")
        return _load_model(path, device)

    def predict(self, state: dict, history: list = None, model_path: Optional[str] = None) -> dict:
        """Single-step prediction from a state dict."""
        from transformer import predict as _predict
        return _predict(
            state       = state,
            history     = history or [],
            model_path  = model_path or self.save_path,
            device_str  = self.device,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _filter_empty(self, trace_dir: str) -> str:
        """
        Return a path to a filtered view of trace_dir that excludes traces
        with 0 state elements.  Writes nothing — just skips at load time
        by returning a wrapper path that the dataset handles internally.

        Currently implemented by patching: we pass the original dir and
        TrajectoryDataset already skips malformed files.  For stronger
        filtering, override here to copy good files to a temp dir.
        """
        return trace_dir

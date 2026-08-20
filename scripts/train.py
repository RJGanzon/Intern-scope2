"""
train.py — Standalone BCTrainer entry point
============================================
Trains (or retrains) the TransformerAgentNetwork from recorded trace sessions.

Usage
-----
    python scripts/train.py                                   # defaults
    python scripts/train.py --trace_dir data/output/traces/live --epochs 50
    python scripts/train.py --epochs 100 --batch_size 32 --device cpu

Arguments
---------
  --trace_dir   Directory containing trace JSONs / session_* sub-dirs
                (default: data/output/traces/live)
  --save_path   Where to write the trained checkpoint
                (default: tasks/form_filling/model.pt)
  --epochs      Training epochs (default: 50)
  --batch_size  Mini-batch size (default: 16)
  --lr          Learning rate (default: 1e-3)
  --val_split   Validation fraction (default: 0.15)
  --aug_drop    Element dropout augmentation probability (default: 0.1)
  --device      auto | cpu | cuda | mps (default: auto)
  --continual   After training, start ContinualLearner to watch for new traces
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys

faulthandler.enable()  # print C-level traceback on segfault/SIGABRT

# ── resolve paths so imports work from the project root ────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMP = os.path.join(_ROOT, "components")
for _p in (_ROOT, _COMP):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def build_parser():
    """The CLI, separated from main() so its defaults can be tested.

    --max_elements in particular is a flag whose wrong value fails silently,
    so it is worth a test that it exists and still defaults to 128.
    """
    parser = argparse.ArgumentParser(
        description="Train Intern's TransformerAgentNetwork via Behavioral Cloning."
    )
    parser.add_argument("--trace_dir",  default="data/output/traces/live",
                        help="Trace directory (flat or session_* sub-dirs). Comma-separate "
                             "multiple directories to pool them into one training run, e.g. "
                             "tasks/form_filling/traces,data/demos/eight_Tabs")
    parser.add_argument("--save_path",  default="tasks/form_filling/model.pt",
                        help="Checkpoint output path")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_split",  type=float, default=0.15)
    parser.add_argument("--aug_drop",   type=float, default=0.1)
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--continual",  action="store_true",
                        help="After training, start ContinualLearner to watch trace_dir")
    # Architecture — shrink these for small datasets to avoid overfitting
    parser.add_argument("--d_model",        type=int,   default=64,
                        help="Transformer hidden size (default: 64, use 128 for large datasets)")
    parser.add_argument("--num_layers",     type=int,   default=2,
                        help="Transformer encoder layers (default: 2)")
    parser.add_argument("--dim_feedforward",type=int,   default=128,
                        help="Feedforward dim (default: 128)")
    parser.add_argument("--dropout",        type=float, default=0.2,
                        help="Dropout rate (default: 0.2)")
    parser.add_argument("--max_elements",   type=int,   default=128,
                        help="Elements per state the model can see. Anything past this "
                             "is dropped: a click landing there resolves to no element "
                             "and teaches nothing. A 50-row grid needs far more than the "
                             "default (the scope #2 grade portal is 303).")
    parser.add_argument("--hist_len",       type=int,   default=4,
                        help="Action-history window the model sees (default: 4; "
                             "try 8-12 for long multi-tab trajectories)")
    parser.add_argument("--disambiguate_attempted", default="none",
                        choices=["none", "rank", "section"],
                        help="Repeated-label ('attempted' feature) disambiguation for Driver 1/2/3-"
                             "style sections (default: none). See transformer.py's TrajectoryDataset "
                             "docstring for the full rationale of each mode.")
    parser.add_argument("--rare_weight_basis", default="type",
                        choices=["none", "type", "field"],
                        help="Rare-action loss up-weighting basis (default: type).")
    parser.add_argument("--section_pattern", default=None,
                        help="Regex (2 groups) identifying section-pane labels, e.g. "
                             "r'section_(driver|vehicle)_(\\d+)$'. Only used with "
                             "--disambiguate_attempted=section.")
    parser.add_argument("--section_prefix", default="section_",
                        help="Prefix identifying section-pane elements by label.")
    parser.add_argument("--action_space", default="legacy",
                        choices=["legacy", "semantic"],
                        help="'legacy' (default) -- unchanged. 'semantic' -- Universal Semantic "
                             "Action Space, ported 2026-08-12 from origin/verb-loop-rewrite. "
                             "Unverified on this project's own data until an isolated A/B runs.")
    return parser


def main():
    args = build_parser().parse_args()

    # Resolve relative paths from the project root — comma-separated --trace_dir
    # pools multiple directories into one training run (see TrajectoryDataset).
    def _resolve(d: str) -> str:
        d = d.strip()
        return d if os.path.isabs(d) else os.path.join(_ROOT, d)

    trace_dirs = [_resolve(d) for d in args.trace_dir.split(",") if d.strip()]
    save_path  = args.save_path if os.path.isabs(args.save_path) \
        else os.path.join(_ROOT, args.save_path)

    for d in trace_dirs:
        if not os.path.isdir(d):
            print(f"[ERROR] Trace directory not found: {d}")
            sys.exit(1)
    trace_dir = trace_dirs if len(trace_dirs) > 1 else trace_dirs[0]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    from intelligence.training.bc.behavioral_cloning import BCTrainer

    trainer = BCTrainer(
        trace_dir=trace_dir,
        save_path=save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        aug_drop=args.aug_drop,
        device=args.device,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        hist_len=args.hist_len,
        max_elements=args.max_elements,
        disambiguate_attempted=args.disambiguate_attempted,
        rare_weight_basis=args.rare_weight_basis,
        section_pattern=args.section_pattern,
        section_prefix=args.section_prefix,
        action_space=args.action_space,
    )

    print(f"\n{'='*60}")
    print(f"  Intern — Behavioral Cloning Trainer")
    print(f"{'='*60}")
    print(f"  trace_dir  : {trace_dir}")
    print(f"  save_path  : {save_path}")
    print(f"  epochs     : {args.epochs}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  device     : {args.device}")
    print(f"{'='*60}\n")

    try:
        trainer.train()
    except Exception as _e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n  Checkpoint saved -> {save_path}")

    if args.continual:
        print("\n  Starting ContinualLearner — watching for new traces …")
        print("  Press Ctrl+C to stop.\n")
        from intelligence.training.continual.learner import ContinualLearner
        cl = ContinualLearner(trace_dir=trace_dir, bc_trainer=trainer)
        cl.start()
        try:
            import time
            while True:
                stats = cl.stats
                print(
                    f"\r  [CL] known={stats['known_traces']}  "
                    f"queued={stats['new_queued']}  "
                    f"retrains={stats['retrain_count']}",
                    end="", flush=True,
                )
                time.sleep(5)
        except KeyboardInterrupt:
            cl.stop()
            print("\n  ContinualLearner stopped.")


if __name__ == "__main__":
    main()

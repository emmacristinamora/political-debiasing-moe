# src/10_run_moce.py


# === IMPORTS ===

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


# === CONFIG ===

# src/09_moce_components.py begins with a digit and cannot be imported via
# normal "import" syntax; we load it explicitly by absolute path.
COMPONENTS_PATH = Path(__file__).resolve().parent / "09_moce_components.py"


# === HELPERS ===

def _load_components_module() -> Any:
    """
    Load src/09_moce_components.py via importlib.

    Returns:
        The loaded module exposing Router and RouterConfig.
    """
    spec = importlib.util.spec_from_file_location("moce_components", COMPONENTS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load components module at {COMPONENTS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["moce_components"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    """
    Parse runtime arguments and enforce the calibrated/checkpoint contract.

    Logic:
        Default mode is heuristic and requires no checkpoint. Calibrated
        mode is opt-in and must be paired with a checkpoint path; passing a
        checkpoint without --calibrated is also rejected as misconfiguration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run MoCE inference. Currently exposes only router construction "
            "and (optionally) calibrated-router checkpoint loading; the rest "
            "of the pipeline is not yet implemented."
        ),
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Enable calibrated routing. Requires --router-checkpoint.",
    )
    parser.add_argument(
        "--router-checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a calibrated-router checkpoint produced by "
            "src/train_calibrated_router.py. Required iff --calibrated."
        ),
    )
    parser.add_argument(
        "--calibration-input-dim",
        "--router-hidden-dim",
        dest="calibration_input_dim",
        type=int,
        default=128,
        help=(
            "Input dimension for the calibrated router correction head. Used "
            "only in calibrated mode; must match the checkpoint's "
            "calibration_input_dim (or legacy router_hidden_dim) or load fails "
            "loudly. --router-hidden-dim is a deprecated alias."
        ),
    )
    args = parser.parse_args()

    # flag/checkpoint contract: fail loudly on mismatched arguments rather
    # than silently dropping into heuristic mode or ignoring a checkpoint
    if args.calibrated and args.router_checkpoint is None:
        parser.error("--calibrated requires --router-checkpoint")
    if (not args.calibrated) and args.router_checkpoint is not None:
        parser.error(
            "--router-checkpoint was provided without --calibrated; "
            "remove the checkpoint or pass --calibrated"
        )
    return args


def build_router(args: argparse.Namespace, moce_components: Any) -> Any:
    """
    Construct Router from CLI args and load a checkpoint in calibrated mode.

    Logic:
        Builds RouterConfig with use_calibrated_router=args.calibrated; in
        calibrated mode, calls Router.load_calibration_checkpoint on the
        provided path. Any FileNotFoundError or ValueError raised by the
        loader (incompatible hidden dim, canonical-order mismatch, missing
        keys, missing file) is propagated unchanged.
    """
    router_config = moce_components.RouterConfig(
        use_calibrated_router=args.calibrated,
        calibration_input_dim=args.calibration_input_dim,
    )
    router = moce_components.Router(router_config)

    if args.calibrated:
        router.load_calibration_checkpoint(args.router_checkpoint)

    return router


def report_router_status(args: argparse.Namespace, router: Any) -> None:
    """
    Print a small, human-readable status line for the constructed router.
    """
    mode = "calibrated" if args.calibrated else "heuristic"
    print(f"router mode:        {mode}")
    if args.calibrated:
        meta = router.calibration_checkpoint_metadata or {}
        print(f"calibration input dim: {router.calibration_input_dim}")
        print(f"checkpoint:         {meta.get('checkpoint_path')}")
        if "beta" in meta:
            print(f"checkpoint beta:    {meta['beta']}")
        if "temperature" in meta:
            print(f"checkpoint temp:    {meta['temperature']}")


# === MAIN ===

def main() -> None:
    args = parse_args()
    moce_components = _load_components_module()
    router = build_router(args, moce_components)
    report_router_status(args, router)

    # the rest of the MoCE pipeline -- InputTransformer, ExpertManager,
    # Editor, MoCEEngine -- is not yet implemented (their methods still
    # raise NotImplementedError). this entrypoint stops here rather than
    # faking that path; once those components land, main() will construct
    # MoCEEngine, hand it the router built above, and call engine.run(...).
    print("(pipeline beyond Router is not yet implemented; nothing further to run)")


if __name__ == "__main__":
    main()

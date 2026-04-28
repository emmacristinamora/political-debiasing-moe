# src/06_moce_architecture.py


# === IMPORTS ===

from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

import torch
from torch import nn


# === CANONICAL ORDER ===

# single source of truth for the four quadrant/expert identities and their
# canonical ordering. use this tuple whenever code needs to move between
# dict-keyed policies and ordered representations (logits, probability
# vectors, diagnostics dumps, per-expert aggregation). all router, editor,
# and expert-manager surfaces must respect these exact keys and this order.
CANONICAL_QUADRANT_ORDER: tuple[str, ...] = (
    "left_lib",
    "left_auth",
    "right_lib",
    "right_auth",
)


# === DATACLASSES ===

@dataclass
class SteeringVectorConfig:
    """
    Configuration for loading and using steering vectors.

    Responsibilities:
    - define where economic and social vector artifacts are stored
    - specify which vector method to use at inference time
    - keep layer and normalization choices explicit so inference matches vector construction

    Notes:
    - this config should stay aligned with the choices made in 03_extract_activations.py
      and 04_build_steering_vectors.py
    - v1 should default to the final aggregated vectors rather than per-layer routing
    """

    economic_vector_path: Path
    social_vector_path: Path
    vector_method: str = "logistic_regression"
    use_final_aggregated_vectors: bool = True
    selected_layers: list[int] = field(default_factory=lambda: [8, 12, 16, 20, 24])
    pooling_method: str = "mean"
    use_centering: bool = False
    neutral_reference_path: Path | None = None


@dataclass
class RouterConfig:
    """
    Configuration for heuristic and calibrated routing.

    Logic:
    - pi_0 is the counterbalancing prior derived from prompt alignment
    - pi is the calibrated router policy defined around pi_0
    - KL anchoring should keep learned routing near the intended debias geometry

    use_calibrated_router semantics:
    - False -> heuristic-only routing (current v1 behavior); Router emits
      pi = pi_0 and does not require a calibration module
    - True  -> calibrated routing; Router emits pi = softmax(log(pi_0) + delta(h)).
      This mode requires a loaded calibration module. If the flag is True
      but no valid module is available, Router must fail loudly rather than
      silently fall back to the heuristic prior.

    Notes:
    - v1 is heuristic-only; only beta, temperature, fallback_to_uniform_if_centered,
      and center_threshold are active
    - kl_weight, entropy_weight, router_hidden_dim, and use_calibrated_router=True
      are placeholders reserved for the future calibrated extension
    """

    use_calibrated_router: bool = False             # v1: keep False; True path is not implemented yet
    beta: float = 1.0                               # v1 active: scales -beta * q_i in heuristic prior
    temperature: float = 1.0                        # v1 active: softmax temperature on the prior logits
    kl_weight: float = 0.1                          # calibrated-mode placeholder, unused in v1
    entropy_weight: float = 0.01                    # calibrated-mode placeholder, unused in v1
    router_hidden_dim: int = 128                    # calibrated-mode placeholder, unused in v1
    fallback_to_uniform_if_centered: bool = True    # v1 active: near-center prompts get uniform prior
    center_threshold: float = 0.05                  # v1 active: threshold on bias_magnitude for fallback


@dataclass
class ExpertConfig:
    """
    Configuration for loading and running quadrant experts.

    Responsibilities:
    - define checkpoint locations for the four pretrained experts
    - specify how expert outputs should be collected
    - keep dense MoE behavior explicit

    Notes:
    - experts must remain separate modules and must not be merged into the base model
    - the checkpoint fields map to CANONICAL_QUADRANT_ORDER
      (left_lib_checkpoint, left_auth_checkpoint, right_lib_checkpoint, right_auth_checkpoint)
    """

    left_lib_checkpoint: Path
    left_auth_checkpoint: Path
    right_lib_checkpoint: Path
    right_auth_checkpoint: Path
    run_dense_moe: bool = True
    return_hidden_states: bool = True
    return_decoded_text: bool = True


@dataclass
class EditorConfig:
    """
    Configuration for recursive fusion and correction.

    Logic:
    - initialize editor weights from the router unless explicitly overridden
    - aggregate expert outputs into a fused hidden-state representation
    - compute correction from ideological alignment of the current mixture
    - update weights and recompute until convergence or max steps

    initialization_mode semantics:
    - "router_policy" (default): initialize alpha from RouterState
      (calibrated_policy when available, otherwise heuristic_prior)
    - "uniform": initialize alpha as a uniform distribution over
      CANONICAL_QUADRANT_ORDER, ignoring the router policy
    Allowed values are exactly {"router_policy", "uniform"}.

    Notes:
    - v1 should default to one update step
    - multi-step recursion should remain available for later experimentation
    """

    max_edit_steps: int = 1
    use_recursive_editing: bool = True
    initialize_from_router: bool = True
    correction_beta: float = 1.0
    convergence_threshold: float = 1e-3
    stop_on_small_weight_change: bool = True
    rescore_current_mixture: bool = True
    keep_edit_trace: bool = True
    initialization_mode: str = "router_policy"


@dataclass
class GenerationConfig:
    """
    Configuration for model generation and decoding.

    Responsibilities:
    - define generation settings shared by all experts
    - keep decoding behavior consistent across router/editor experiments
    """

    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = False
    top_p: float = 1.0


@dataclass
class PromptState:
    """
    Political-state representation of a prompt.

    Contains:
    - prompt text
    - hidden representation used for projections
    - axis scores in compass space
    - canonical quadrant scores
    - bias magnitude or distance from center

    This object is the output of InputTransformer and the input to Router.

    Router input contract (v1, heuristic):
    - active routing inputs: quadrant_scores, bias_magnitude
    - diagnostics only (not primary routing signal): economic_score, social_score
    - carried for future calibrated routing only: hidden_representation
    - traceability only (not a heuristic routing signal): prompt_text, metadata

    quadrant_scores keys: see CANONICAL_QUADRANT_ORDER (module-level constant).

    hidden_representation contract:
    - heuristic routing does NOT consume hidden_representation; it may be
      omitted or carried purely for diagnostics in heuristic mode.
    - calibrated routing (use_calibrated_router=True) REQUIRES it. When
      required, hidden_representation must be:
        - a 1D numeric vector with shape [hidden_dim]
        - finite (no NaN, no inf)
        - drawn from the same base model, the same layer (or layer
          aggregation), and the same token-pooling strategy used to
          extract steering vectors and to train the calibration module
    - the field is intentionally typed as Any at this step to keep the
      backbone choice (e.g., torch.Tensor vs np.ndarray) flexible; the
      contract above is enforced by Router in calibrated mode, not by
      the dataclass itself.
    - dimensional consistency: hidden_dim must match the input dimension
      expected by the loaded calibration module; a mismatch must surface
      as a runtime error at routing time, not be silently broadcast or
      truncated.
    """

    prompt_text: str
    hidden_representation: Any
    economic_score: float
    social_score: float
    quadrant_scores: dict[str, float]
    bias_magnitude: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterState:
    """
    Routing outputs used downstream by the editor.

    Contains:
    - heuristic prior pi_0
    - calibrated router policy pi
    - optional training losses or diagnostics

    Router output contract:
    - heuristic_prior: normalized distribution over CANONICAL_QUADRANT_ORDER, sums to 1
    - calibrated_policy: normalized distribution over the same key set;
      in heuristic-only mode it equals heuristic_prior exactly;
      in calibrated mode it equals softmax(log(pi_0) + delta(h)), where
      delta(h) is produced by the loaded calibration module from
      PromptState.hidden_representation. When delta(h) is zero
      elementwise, calibrated_policy collapses back to heuristic_prior.
    - diagnostics: trace data keyed by "beta", "temperature",
      "used_center_fallback", "quadrant_scores" (copy), "heuristic_prior" (copy)
    - losses: empty dict in heuristic-only mode; in calibrated mode it
      may carry router regularization terms (KL anchor, entropy) for
      reporting only -- Router does not optimize them at inference time

    Notes:
    - downstream editor consumes this object directly
    - when serializing to an ordered vector, iterate CANONICAL_QUADRANT_ORDER
    """

    heuristic_prior: dict[str, float]
    calibrated_policy: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    losses: dict[str, float] = field(default_factory=dict)


@dataclass
class ExpertOutput:
    """
    Unified representation of a single expert response.

    Contains:
    - expert name
    - hidden-state output for editor-side fusion
    - optional decoded text for logging or fallback synthesis
    - metadata for debugging
    """

    expert_name: str
    hidden_output: Any | None = None
    decoded_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditorStepTrace:
    """
    Trace of one editor iteration.

    Contains:
    - step index
    - alpha (mixture weights) before and after the correction update
    - delta correction logits applied at this step
    - mixture alignment before and after re-aggregation
    - scalar summaries of weight and alignment movement, useful for
      stop diagnostics and post-hoc inspection

    Notes:
    - this trace is intentionally lightweight: dict-of-float scalars only.
      Do not store hidden-state tensors or decoded text here.
    """

    step_index: int
    alpha_before: dict[str, float]
    delta: dict[str, float]
    alpha_after: dict[str, float]
    alignment_before: dict[str, float]
    alignment_after: dict[str, float]
    max_alpha_change: float
    max_alignment_change: float


@dataclass
class EditorResult:
    """
    Full output of Editor.run_editing_loop.

    Contains:
    - final fused hidden state ready for downstream decoding
    - final mixture weights (alpha) and ideological alignment
    - per-step traces for interpretability
    - run-level metadata: number of steps run, early-stop flag, stop reason

    Notes:
    - the editor returns hidden-state mixing artifacts only; decoding the
      final hidden state into text is owned by MoCEEngine.run
    - stop_reason is None when the loop exited because max_edit_steps was
      reached without an early-stop condition firing; otherwise it names
      the condition that triggered early termination (e.g.
      "small_alpha_change", "small_alignment_change")
    """

    final_mixed_hidden_state: torch.Tensor
    final_alpha: dict[str, float]
    final_alignment: dict[str, float]
    step_traces: list[EditorStepTrace]
    num_steps_run: int
    stopped_early: bool
    stop_reason: str | None = None


@dataclass
class MoCEResult:
    """
    Full output of one MoCE run.

    This should be rich enough that 07_run_moce.py only needs to save it,
    not reconstruct anything after the fact.

    Ownership boundary:
    - Editor returns mixing artifacts (see EditorResult); it does not decode.
    - MoCEEngine.run owns downstream decoding and packages the final text
      into this result.
    """

    prompt_text: str
    prompt_state: PromptState
    router_state: RouterState
    expert_outputs: dict[str, ExpertOutput]
    editor_result: EditorResult | None
    final_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# === INPUT TRANSFORMER ===

class InputTransformer:
    """
    Project prompts into political-compass space.

    Responsibilities:
    - encode prompt hidden states using the base backbone
    - project onto steering vectors for economic and social axes
    - derive canonical quadrant scores for routing

    Returns:
    - a structured PromptState containing all ideological diagnostics

    Important:
    - inference-time representations must match the hidden-state space used
      when the steering vectors were learned
    - this component is not generic preprocessing; it is the prompt-state estimator
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        steering_config: SteeringVectorConfig,
    ) -> None:
        # store base model, tokenizer, and steering-vector settings
        # load economic/social vectors and any optional centering reference
        raise NotImplementedError

    def load_steering_vectors(self) -> None:
        """
        Load economic and social steering-vector artifacts from disk.

        Logic:
        - support both final aggregated vectors and per-layer vectors
        - validate that vector metadata matches inference assumptions
        """
        raise NotImplementedError

    def encode_prompt(self, prompt_text: str) -> Any:
        """
        Encode prompt into the same hidden-state space used to build steering vectors.

        Logic:
        - run the prompt through the base model
        - extract hidden states from the selected layers
        - pool token representations according to the configured pooling method
        """
        raise NotImplementedError

    def maybe_center_representation(self, hidden_representation: Any) -> Any:
        """
        Optionally subtract a neutral reference representation before projection.

        Logic:
        - use centering only if a neutral reference has been explicitly configured
        - keep both centered and uncentered behavior easy to inspect
        """
        raise NotImplementedError

    def compute_axis_scores(self, hidden_representation: Any) -> dict[str, float]:
        """
        Compute signed projections on economic and social axes.

        Returns:
        - dictionary with economic_score and social_score
        """
        raise NotImplementedError

    def compute_quadrant_scores(self, hidden_representation: Any) -> dict[str, float]:
        """
        Derive canonical quadrant affinities from political-compass directions.

        Logic:
        - compute scores for left_lib, left_auth, right_lib, right_auth
        - use the canonical quadrant vectors built from signed axis combinations
        """
        raise NotImplementedError

    def compute_bias_magnitude(
        self,
        economic_score: float,
        social_score: float,
    ) -> float:
        """
        Compute distance from political center in compass space.

        Notes:
        - this is useful for routing fallback behavior and later diagnostics
        """
        raise NotImplementedError

    def transform(self, prompt_text: str) -> PromptState:
        """
        Full input-transformation pipeline.

        Flow:
        - encode prompt
        - optionally center representation
        - compute axis scores
        - compute quadrant scores
        - package everything into PromptState
        """
        raise NotImplementedError


# === CALIBRATED ROUTER TRAINING DATASET SCHEMA ===

# documentation-only schema for the calibration module's training data. this
# block defines the on-disk artifact format expected by a future standalone
# training script; no loader, validator, or trainer is implemented in this
# module. inference-time routing does not depend on this schema.
#
# recommended artifact layout
# ---------------------------
# a calibrated-router dataset is split across two kinds of files in the same
# directory:
#
#   1. records.jsonl  -- one JSON object per training example; lightweight
#                        structured fields only, no raw dense vectors
#   2. hidden.pt      -- a torch tensor artifact holding the dense hidden
#                        representations referenced from records.jsonl;
#                        recommended shape [num_examples, hidden_dim] with
#                        dtype float32. exact filename and tensor format are
#                        writer-defined; .pt is the recommended default.
#
# the split keeps JSONL small, human-readable, and grep-able, while heavy
# float vectors live in a compact binary artifact. JSONL stores references;
# tensor files store the actual feature vectors. inlining dense vectors in
# JSONL is explicitly NOT supported.
#
# per-example record schema (one line in records.jsonl)
# -----------------------------------------------------
#   example_id:                 str
#       stable unique id for the example. used for deduplication and to
#       cross-reference logs and downstream evaluations.
#
#   prompt_text:                str
#       original prompt text. carried for traceability; not consumed as a
#       training feature.
#
#   quadrant_scores:            dict[str, float]
#       prompt-side alignment scores feeding the heuristic prior. keys must
#       be exactly CANONICAL_QUADRANT_ORDER (any insertion order); values
#       must be finite floats. these are the same scores the heuristic
#       router consumes at inference time.
#
#   bias_magnitude:             float
#       distance from political center in compass space; finite scalar.
#       used by the heuristic prior's center-fallback gate.
#
#   target_policy:              dict[str, float]
#       supervision signal: the desired calibrated policy for this example.
#       keys must be exactly CANONICAL_QUADRANT_ORDER. values must form a
#       valid probability distribution: strictly positive and summing to 1
#       within floating-point tolerance. how target_policy is produced
#       (counterbalancing rule, teacher model, hand-curated, etc.) is out
#       of scope for this schema.
#
#   hidden_representation_ref:  str
#       pointer to the dense feature vector for this example. resolution is
#       deterministic and writer-defined; the recommended form is
#       "<filename>:<row_index>" (e.g. "hidden.pt:42"), referencing a row in
#       a 2D tensor of shape [num_examples, hidden_dim]. the referenced
#       vector must come from the same base model, layer (or layer
#       aggregation), and token-pooling strategy used both for steering-
#       vector extraction and for runtime calibrated routing -- this is the
#       same activation-space contract documented on
#       PromptState.hidden_representation.
#
#   metadata:                   dict[str, Any]   (optional; default {})
#       free-form provenance fields (source corpus id, generation timestamp,
#       teacher model version, etc.). not consumed by the trainer; kept for
#       reproducibility only.
#
# per-example validation expectations (to be enforced by the training script)
# ---------------------------------------------------------------------------
#   - all required fields are present and have the declared types
#   - quadrant_scores keys are exactly CANONICAL_QUADRANT_ORDER and values
#     are finite floats
#   - target_policy keys are exactly CANONICAL_QUADRANT_ORDER, values are
#     finite, strictly positive, and sum to 1 within ~1e-6
#   - bias_magnitude is a finite float
#   - hidden_representation_ref resolves to an existing 1D vector in the
#     associated tensor artifact (no missing rows, no broken pointers)
#   - the resolved hidden vector's length equals the calibration module's
#     declared input dimension (RouterConfig.router_hidden_dim used at
#     inference time); a mismatch is a hard error, not a silent reshape,
#     pad, or truncation -- mirrors the runtime contract enforced by
#     Router._prepare_hidden_representation
#
# out of scope for this schema
# ----------------------------
# these are deliberately NOT part of a training record:
#   - expert hidden-state outputs or expert-generated text
#   - editor traces, weight updates, or mixture alignments
#   - MoCEEngine final generated text
#   - per-example training losses (computed at training time, not stored)
#   - inline raw dense vectors in JSONL (always go through
#     hidden_representation_ref)
#
# practical scope
# ---------------
# one clean format is sufficient for v1. no competing schemas, no nested
# task formats, no versioning framework, no sharding rules. add complexity
# only when an actual requirement emerges.


# === ROUTER ===

class Router:
    """
    Compute initial expert routing for debiasing.

    Scope (v1):
    - heuristic-only: deterministic pi_0 = softmax(-beta * q / temperature),
      with optional uniform fallback for near-center prompts
    - calibrated methods (compute_router_correction, combine_prior_and_correction,
      compute_router_losses) are part of the interface but unimplemented;
      route() raises NotImplementedError when use_calibrated_router=True
    - consumes precomputed prompt geometry from PromptState; never runs a
      model forward pass

    Calibrated routing (definition; not yet implemented):
    - pi_0     : heuristic prior built from quadrant scores (v1 logic)
    - delta(h) : learned correction logits derived from the prompt's
                 hidden representation
    - pi       : final calibrated policy
    - formula  : pi = softmax(log(pi_0) + delta(h))
    - semantics: calibrated routing does NOT replace the heuristic prior;
                 it modifies it additively in log-space.

    Calibration module (interface; not yet implemented):
    - a learned correction module is owned by Router (not by an external
      component) and lives inside this class
    - input  : PromptState.hidden_representation
    - output : 4 logits aligned exactly with CANONICAL_QUADRANT_ORDER;
               no alternative ordering is permitted at any boundary
    - architecture (linear vs MLP) is intentionally left abstract here;
      it is a single learned mapping h -> R^4

    Runtime behavior (use_calibrated_router):
    - False : heuristic-only routing; pi = pi_0; calibrated_policy mirrors
              heuristic_prior; no correction module is required or loaded
    - True  : calibrated routing; pi = softmax(log(pi_0) + delta(h));
              requires a loaded calibration module. If enabled without a
              valid module, Router must fail loudly (not silently fall back
              to the heuristic prior).

    Artifact boundary:
    - calibration weights are persisted separately from the base model and
      from the heuristic configuration; they are loaded at runtime only
      when calibrated mode is enabled.
    - a calibration checkpoint contains:
        - the learned correction module weights
        - minimal metadata (e.g., input dimension, canonical ordering)
          sufficient to validate that the module matches
          CANONICAL_QUADRANT_ORDER and PromptState.hidden_representation
          at load time
    - exact file paths and serialization formats are out of scope here.

    Training vs inference separation:
    - training of the calibrated router lives in a separate script and is
      out of scope for Router. Router is responsible only for inference:
      producing delta(h), combining it with pi_0, and reporting losses
      for diagnostics.
    - compute_router_losses returns regularization terms (KL anchor,
      entropy) for inspection only; no optimization happens inside Router.

    Input contract:
    - treat prompt_state.quadrant_scores as authoritative input geometry
    - do not recompute quadrants from economic_score / social_score
    - do not use prompt_text for routing
    - in calibrated mode, prompt_state.hidden_representation is the sole
      input to the correction module

    hidden_representation usage (calibrated mode):
    - consumed directly by the correction module; Router does not
      recompute, re-pool, re-normalize, or otherwise transform it
    - assumed to be produced upstream by InputTransformer and to satisfy
      the PromptState.hidden_representation contract (1D, numeric, finite,
      shape [hidden_dim]) drawn from the same model/layer/pooling used
      during steering-vector extraction and calibration training
    - dimensional consistency is mandatory: hidden_dim must match the
      input dimension declared by the loaded calibration module's
      checkpoint metadata; any mismatch must raise an error at runtime
      rather than be silently reshaped, padded, or truncated

    Calibrated-mode validation requirement (to be implemented later in
    compute_router_correction; not implemented in this step):
    - hidden_representation must be present (not None)
    - it must be a 1D numeric vector
    - all entries must be finite (no NaN, no inf)
    - its dimension must match the calibration module's expected input
    - any violation must raise ValueError with a precise message
      identifying the failed condition (presence / shape / finiteness /
      dimensional mismatch)
    - heuristic-mode routing must NOT trigger any of these checks

    Output contract:
    - policies are normalized dicts keyed by CANONICAL_QUADRANT_ORDER
    - iterate CANONICAL_QUADRANT_ORDER when converting to/from ordered logits
    - key set stays aligned with ExpertConfig / ExpertManager naming

    Invariants:
    - if delta(h) is zero elementwise, calibrated policy equals the
      heuristic prior exactly: pi == pi_0
    - the output is always a valid probability distribution over the four
      quadrants (non-negative entries that sum to 1)
    - canonical quadrant ordering is preserved end-to-end, from
      PromptState.quadrant_scores through delta(h) logits to the keys of
      the final pi dict

    Important:
    - prompts near a quadrant downweight that quadrant and upweight the
      opposite and adjacent quadrants
    - the calibrated router, when implemented, learns a small correction
      around the heuristic prior, not a free policy from scratch
    """

    def __init__(self, config: RouterConfig) -> None:
        # store router hyperparameters; calibration setup happens below only in calibrated mode
        self.config = config

        if not config.use_calibrated_router:
            # heuristic mode: keep calibration attributes inert so downstream code
            # can introspect them uniformly without branching on the flag
            self.calibration_module: nn.Module | None = None
            self.calibration_input_dim: int | None = None
            self.calibration_checkpoint_metadata: dict[str, Any] | None = None
            return

        # calibrated mode: router_hidden_dim is reinterpreted here as the
        # input dimension expected by the calibration module (i.e. the
        # dimensionality of PromptState.hidden_representation). a dedicated
        # field will replace this overload in a later step.
        router_hidden_dim = config.router_hidden_dim
        if not isinstance(router_hidden_dim, int) or isinstance(router_hidden_dim, bool):
            raise ValueError(
                "RouterConfig.router_hidden_dim must be a positive int when "
                f"use_calibrated_router=True; got {type(router_hidden_dim).__name__}"
            )
        if router_hidden_dim <= 0:
            raise ValueError(
                "RouterConfig.router_hidden_dim must be a positive int when "
                f"use_calibrated_router=True; got {router_hidden_dim}"
            )

        self.calibration_input_dim: int | None = router_hidden_dim
        self.calibration_module: nn.Module | None = nn.Linear(
            router_hidden_dim,
            len(CANONICAL_QUADRANT_ORDER),
        )
        # populated by load_calibration_checkpoint(); None until a checkpoint
        # is loaded, even in calibrated mode
        self.calibration_checkpoint_metadata: dict[str, Any] | None = None

    def _validate_prompt_state(self, prompt_state: PromptState) -> None:
        """
        Fail-fast validation of router inputs.

        Logic:
        - quadrant_scores must be a dict with exactly CANONICAL_QUADRANT_ORDER keys
        - every quadrant score must be a finite int/float
        - bias_magnitude must be a finite int/float

        Raises:
        - ValueError on any malformed routing input
        """
        quadrant_scores = prompt_state.quadrant_scores
        if quadrant_scores is None:
            raise ValueError(
                "PromptState.quadrant_scores is None; "
                f"expected a dict over {list(CANONICAL_QUADRANT_ORDER)}"
            )
        if not isinstance(quadrant_scores, dict):
            raise ValueError(
                "PromptState.quadrant_scores must be a dict, "
                f"got {type(quadrant_scores).__name__}"
            )

        expected_keys = set(CANONICAL_QUADRANT_ORDER)
        actual_keys = set(quadrant_scores.keys())
        missing_keys = expected_keys - actual_keys
        if missing_keys:
            raise ValueError(
                f"PromptState.quadrant_scores is missing required keys: {sorted(missing_keys)}; "
                f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
            )
        unexpected_keys = actual_keys - expected_keys
        if unexpected_keys:
            raise ValueError(
                f"PromptState.quadrant_scores has unexpected keys: {sorted(unexpected_keys)}; "
                f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
            )

        for key in CANONICAL_QUADRANT_ORDER:
            value = quadrant_scores[key]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"PromptState.quadrant_scores[{key!r}] must be int or float, "
                    f"got {type(value).__name__}"
                )
            if math.isnan(value):
                raise ValueError(f"PromptState.quadrant_scores[{key!r}] is NaN")
            if math.isinf(value):
                raise ValueError(f"PromptState.quadrant_scores[{key!r}] is infinite")

        bias_magnitude = prompt_state.bias_magnitude
        if not isinstance(bias_magnitude, (int, float)):
            raise ValueError(
                "PromptState.bias_magnitude must be int or float, "
                f"got {type(bias_magnitude).__name__}"
            )
        if math.isnan(bias_magnitude):
            raise ValueError("PromptState.bias_magnitude is NaN")
        if math.isinf(bias_magnitude):
            raise ValueError("PromptState.bias_magnitude is infinite")

    def _extract_ordered_quadrant_scores(self, prompt_state: PromptState) -> list[float]:
        """
        Return quadrant scores as a list ordered by CANONICAL_QUADRANT_ORDER.

        Logic:
        - validate inputs via _validate_prompt_state
        - read prompt_state.quadrant_scores in canonical order
        """
        self._validate_prompt_state(prompt_state)
        return [float(prompt_state.quadrant_scores[key]) for key in CANONICAL_QUADRANT_ORDER]

    def _softmax(self, logits: list[float]) -> list[float]:
        """
        Numerically stable softmax over a list of logits.

        Logic:
        - validate input list (non-empty, finite numeric values)
        - subtract max(logits) before exponentiation for stability
        - normalize exponentials by their sum
        """
        if len(logits) == 0:
            raise ValueError("_softmax received an empty logits list")
        for index, value in enumerate(logits):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"_softmax logits[{index}] must be int or float, "
                    f"got {type(value).__name__}"
                )
            if math.isnan(value):
                raise ValueError(f"_softmax logits[{index}] is NaN")
            if math.isinf(value):
                raise ValueError(f"_softmax logits[{index}] is infinite")

        max_logit = max(logits)
        shifted_exps = [math.exp(value - max_logit) for value in logits]
        total = sum(shifted_exps)
        return [exp_value / total for exp_value in shifted_exps]

    def _should_use_center_fallback(self, prompt_state: PromptState) -> bool:
        """
        Decide whether to fall back to a uniform prior for near-center prompts.

        Logic:
        - validate inputs via _validate_prompt_state
        - return True only when the gate is enabled and bias_magnitude
          is strictly below the configured center_threshold
        """
        self._validate_prompt_state(prompt_state)
        if not self.config.fallback_to_uniform_if_centered:
            return False
        return prompt_state.bias_magnitude < self.config.center_threshold

    def build_heuristic_prior(self, prompt_state: PromptState) -> dict[str, float]:
        """
        Build heuristic prior pi_0 from quadrant alignment scores.

        Logic:
        - if the prompt is near center, return a uniform prior
        - otherwise compute pi_0 = softmax(-beta * q / temperature) over
          CANONICAL_QUADRANT_ORDER

        Raises:
        - ValueError if RouterConfig.temperature == 0
        """
        if self._should_use_center_fallback(prompt_state):
            uniform_weight = 1.0 / len(CANONICAL_QUADRANT_ORDER)
            return {key: uniform_weight for key in CANONICAL_QUADRANT_ORDER}

        if self.config.temperature == 0:
            raise ValueError(
                "RouterConfig.temperature must be non-zero for heuristic prior; "
                f"got {self.config.temperature}"
            )

        ordered_scores = self._extract_ordered_quadrant_scores(prompt_state)
        logits = [
            -self.config.beta * score / self.config.temperature
            for score in ordered_scores
        ]
        probabilities = self._softmax(logits)
        return {key: prob for key, prob in zip(CANONICAL_QUADRANT_ORDER, probabilities)}

    def _prepare_hidden_representation(self, hidden_representation: Any) -> torch.Tensor:
        """
        Validate hidden_representation and return a 1D float32 tensor.

        Logic:
        - presence: must not be None
        - type:     accept torch.Tensor, list, or tuple of numeric scalars
        - shape:    must be 1D with length == self.calibration_input_dim
        - values:   all entries must be finite (no NaN, no inf)

        Returns:
        - torch.Tensor of dtype float32 and shape [calibration_input_dim]

        Raises:
        - ValueError naming the violated condition; no silent coercion of
          shape, dtype, or value
        """
        if hidden_representation is None:
            raise ValueError(
                "PromptState.hidden_representation is None; "
                "calibrated routing requires a 1D numeric vector"
            )

        expected_dim = self.calibration_input_dim

        if isinstance(hidden_representation, torch.Tensor):
            if hidden_representation.dim() != 1:
                raise ValueError(
                    "PromptState.hidden_representation must be a 1D tensor; "
                    f"got shape {tuple(hidden_representation.shape)}"
                )
            if hidden_representation.shape[0] != expected_dim:
                raise ValueError(
                    f"PromptState.hidden_representation length {hidden_representation.shape[0]} "
                    f"does not match calibration_input_dim {expected_dim}"
                )
            tensor = hidden_representation.to(dtype=torch.float32)
            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    "PromptState.hidden_representation contains NaN or inf entries"
                )
            return tensor

        if isinstance(hidden_representation, (list, tuple)):
            for index, value in enumerate(hidden_representation):
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"PromptState.hidden_representation[{index}] must be int or float, "
                        f"got {type(value).__name__}"
                    )
                if math.isnan(value):
                    raise ValueError(
                        f"PromptState.hidden_representation[{index}] is NaN"
                    )
                if math.isinf(value):
                    raise ValueError(
                        f"PromptState.hidden_representation[{index}] is infinite"
                    )
            if len(hidden_representation) != expected_dim:
                raise ValueError(
                    f"PromptState.hidden_representation length {len(hidden_representation)} "
                    f"does not match calibration_input_dim {expected_dim}"
                )
            return torch.tensor(hidden_representation, dtype=torch.float32)

        raise ValueError(
            "PromptState.hidden_representation must be a torch.Tensor, list, or tuple "
            f"of numeric scalars; got {type(hidden_representation).__name__}"
        )

    def compute_router_correction(self, prompt_state: PromptState) -> dict[str, float]:
        """
        Compute the calibrated correction delta(h) around log(pi_0).

        Logic:
        - require an initialized calibration module
        - validate and convert prompt_state.hidden_representation to a 1D
          float32 tensor of length self.calibration_input_dim
        - run the calibration module to obtain 4 logits aligned with
          CANONICAL_QUADRANT_ORDER

        Returns:
        - dict[str, float] mapping each canonical quadrant key to its
          correction logit (plain Python floats, not tensors)

        Raises:
        - ValueError if calibrated routing was requested but no calibration
          module is initialized, or if hidden_representation violates the
          contract (presence / type / shape / finiteness / dimension)
        """
        if self.calibration_module is None:
            raise ValueError(
                "compute_router_correction requires a calibration module; "
                "none is initialized (use_calibrated_router=False at construction time)"
            )

        hidden_tensor = self._prepare_hidden_representation(prompt_state.hidden_representation)
        logits = self.calibration_module(hidden_tensor)

        if logits.dim() != 1 or logits.shape[0] != len(CANONICAL_QUADRANT_ORDER):
            raise ValueError(
                f"calibration module produced logits of shape {tuple(logits.shape)}; "
                f"expected ({len(CANONICAL_QUADRANT_ORDER)},)"
            )

        return {
            key: float(logits[index].item())
            for index, key in enumerate(CANONICAL_QUADRANT_ORDER)
        }

    def _validate_canonical_quadrant_dict(
        self,
        distribution: Any,
        field_name: str,
        *,
        require_positive: bool,
        require_sums_to_one: bool,
        sum_tolerance: float = 1e-6,
    ) -> None:
        """
        Validate a dict keyed exactly by CANONICAL_QUADRANT_ORDER.

        Logic:
        - dict-ness, complete and exclusive key set, numeric and finite values
        - optional strict positivity (required for log-of-prior)
        - optional sum-to-one within sum_tolerance (required for distributions)

        Raises:
        - ValueError naming the violated condition; no silent normalization
        """
        if not isinstance(distribution, dict):
            raise ValueError(
                f"{field_name} must be a dict, got {type(distribution).__name__}"
            )

        expected_keys = set(CANONICAL_QUADRANT_ORDER)
        actual_keys = set(distribution.keys())
        missing_keys = expected_keys - actual_keys
        if missing_keys:
            raise ValueError(
                f"{field_name} is missing required keys: {sorted(missing_keys)}; "
                f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
            )
        unexpected_keys = actual_keys - expected_keys
        if unexpected_keys:
            raise ValueError(
                f"{field_name} has unexpected keys: {sorted(unexpected_keys)}; "
                f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
            )

        for key in CANONICAL_QUADRANT_ORDER:
            value = distribution[key]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{field_name}[{key!r}] must be int or float, "
                    f"got {type(value).__name__}"
                )
            if math.isnan(value):
                raise ValueError(f"{field_name}[{key!r}] is NaN")
            if math.isinf(value):
                raise ValueError(f"{field_name}[{key!r}] is infinite")
            if require_positive and value <= 0:
                raise ValueError(
                    f"{field_name}[{key!r}] must be strictly positive for log; got {value}"
                )

        if require_sums_to_one:
            total = sum(float(distribution[key]) for key in CANONICAL_QUADRANT_ORDER)
            if abs(total - 1.0) > sum_tolerance:
                raise ValueError(
                    f"{field_name} must sum to 1 within {sum_tolerance}; got sum={total}"
                )

    def combine_prior_and_correction(
        self,
        heuristic_prior: dict[str, float],
        correction_logits: dict[str, float],
    ) -> dict[str, float]:
        """
        Combine heuristic prior and correction into calibrated policy pi.

        Logic:
        - pi = softmax(log(pi_0) + delta(h))
        - validate both inputs strictly; no silent repair
        - read both dicts in CANONICAL_QUADRANT_ORDER, build combined logits,
          run them through _softmax, and return a canonically-ordered dict

        Returns:
        - dict[str, float] with keys exactly CANONICAL_QUADRANT_ORDER and
          values in [0, 1] summing to 1 (up to floating-point tolerance)

        Raises:
        - ValueError if heuristic_prior is not a strictly-positive distribution
          summing to 1 over CANONICAL_QUADRANT_ORDER, or if correction_logits
          is not a finite numeric dict over the same key set
        """
        self._validate_canonical_quadrant_dict(
            heuristic_prior,
            "heuristic_prior",
            require_positive=True,
            require_sums_to_one=True,
        )
        self._validate_canonical_quadrant_dict(
            correction_logits,
            "correction_logits",
            require_positive=False,
            require_sums_to_one=False,
        )

        combined_logits = [
            math.log(float(heuristic_prior[key])) + float(correction_logits[key])
            for key in CANONICAL_QUADRANT_ORDER
        ]
        probabilities = self._softmax(combined_logits)
        return {key: prob for key, prob in zip(CANONICAL_QUADRANT_ORDER, probabilities)}

    def compute_router_losses(
        self,
        heuristic_prior: dict[str, float],
        calibrated_policy: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute router regularization losses.

        Logic:
        - validate both distributions over CANONICAL_QUADRANT_ORDER
        - kl       = sum_i pi_i * (log(pi_i) - log(pi_0_i))   # KL(pi || pi_0)
        - entropy  = -sum_i pi_i * log(pi_i)                  # raw H(pi)

        Returns:
        - {"kl": float, "entropy": float}; entropy is returned raw (not
          negated, not weighted) so training code can decide its sign and
          combine it with config-driven weights into the total loss

        Raises:
        - ValueError if either input is not a strictly-positive distribution
          summing to 1 over CANONICAL_QUADRANT_ORDER
        """
        # both inputs must be valid probability distributions; strict positivity
        # makes math.log safe without epsilon smoothing
        self._validate_canonical_quadrant_dict(
            heuristic_prior,
            "heuristic_prior",
            require_positive=True,
            require_sums_to_one=True,
        )
        self._validate_canonical_quadrant_dict(
            calibrated_policy,
            "calibrated_policy",
            require_positive=True,
            require_sums_to_one=True,
        )

        kl = 0.0
        entropy = 0.0
        for key in CANONICAL_QUADRANT_ORDER:
            pi_i = float(calibrated_policy[key])
            pi_0_i = float(heuristic_prior[key])
            log_pi_i = math.log(pi_i)
            kl += pi_i * (log_pi_i - math.log(pi_0_i))
            entropy += -pi_i * log_pi_i

        return {"kl": float(kl), "entropy": float(entropy)}

    def _validate_calibration_checkpoint_metadata(
        self,
        checkpoint: Any,
        checkpoint_path: Path,
    ) -> None:
        """
        Fail-fast validation of a loaded calibration checkpoint.

        Logic:
        - the checkpoint payload must be a dict
        - required keys: state_dict, router_hidden_dim, canonical_quadrant_order
        - router_hidden_dim must equal self.calibration_input_dim
        - canonical_quadrant_order must equal CANONICAL_QUADRANT_ORDER exactly
          (as a tuple, in canonical order)

        Raises:
        - ValueError naming the missing key or the mismatched field
        """
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"calibration checkpoint at {checkpoint_path} must be a dict, "
                f"got {type(checkpoint).__name__}"
            )
        for key in ("state_dict", "router_hidden_dim", "canonical_quadrant_order"):
            if key not in checkpoint:
                raise ValueError(
                    f"calibration checkpoint at {checkpoint_path} is missing "
                    f"required key {key!r}"
                )

        ckpt_dim = checkpoint["router_hidden_dim"]
        if ckpt_dim != self.calibration_input_dim:
            raise ValueError(
                f"calibration checkpoint at {checkpoint_path} declares "
                f"router_hidden_dim={ckpt_dim}, but this router was constructed "
                f"with calibration_input_dim={self.calibration_input_dim}"
            )

        ckpt_order = tuple(checkpoint["canonical_quadrant_order"])
        if ckpt_order != CANONICAL_QUADRANT_ORDER:
            raise ValueError(
                f"calibration checkpoint at {checkpoint_path} canonical_quadrant_order "
                f"{list(ckpt_order)} does not match CANONICAL_QUADRANT_ORDER "
                f"{list(CANONICAL_QUADRANT_ORDER)}"
            )

    def load_calibration_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load a trained calibration head checkpoint into self.calibration_module.

        Logic:
        - precondition: the router was constructed in calibrated mode
          (self.calibration_module is not None); raises ValueError otherwise.
          this method does not silently initialize a calibration module.
        - reads the checkpoint via torch.load on the given path
        - validates state_dict, router_hidden_dim, and canonical_quadrant_order
          via _validate_calibration_checkpoint_metadata
        - calls calibration_module.load_state_dict(checkpoint["state_dict"])
        - records minimal traceability in self.calibration_checkpoint_metadata

        Args:
        - checkpoint_path: pathlib.Path to a .pt file produced by
          src/train_calibrated_router.py

        Raises:
        - ValueError if the router is in heuristic mode, the checkpoint is
          malformed, or its metadata does not match the router's configuration
        - FileNotFoundError if checkpoint_path does not exist
        """
        if self.calibration_module is None:
            raise ValueError(
                "load_calibration_checkpoint requires a calibration module; "
                "this router was constructed with use_calibrated_router=False"
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"calibration checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self._validate_calibration_checkpoint_metadata(checkpoint, checkpoint_path)
        self.calibration_module.load_state_dict(checkpoint["state_dict"])

        metadata: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "router_hidden_dim": checkpoint["router_hidden_dim"],
            "canonical_quadrant_order": list(checkpoint["canonical_quadrant_order"]),
        }
        if "beta" in checkpoint:
            metadata["beta"] = checkpoint["beta"]
        if "temperature" in checkpoint:
            metadata["temperature"] = checkpoint["temperature"]
        self.calibration_checkpoint_metadata = metadata

    def route(self, prompt_state: PromptState) -> RouterState:
        """
        Full routing pipeline.

        Flow:
        - validate prompt_state
        - build heuristic prior pi_0
        - in heuristic mode, set calibrated_policy = pi_0 and losses = {}
        - in calibrated mode, compute delta(h), combine with pi_0 to get pi,
          and compute regularization losses (KL anchor + raw entropy)
        - populate diagnostics with: beta, temperature, used_center_fallback,
          quadrant_scores (copy), heuristic_prior (copy); in calibrated mode
          additionally include correction_logits (copy) and calibrated_policy
          (copy)

        Raises:
        - ValueError propagated from compute_router_correction /
          combine_prior_and_correction / compute_router_losses (e.g. invalid
          hidden_representation, missing calibration module, malformed
          distributions)
        """
        self._validate_prompt_state(prompt_state)
        heuristic_prior = self.build_heuristic_prior(prompt_state)

        diagnostics = {
            "beta": self.config.beta,
            "temperature": self.config.temperature,
            "used_center_fallback": self._should_use_center_fallback(prompt_state),
            "quadrant_scores": dict(prompt_state.quadrant_scores),
            "heuristic_prior": dict(heuristic_prior),
        }

        if not self.config.use_calibrated_router:
            calibrated_policy = dict(heuristic_prior)
            return RouterState(
                heuristic_prior=heuristic_prior,
                calibrated_policy=calibrated_policy,
                diagnostics=diagnostics,
                losses={},
            )

        correction_logits = self.compute_router_correction(prompt_state)
        calibrated_policy = self.combine_prior_and_correction(
            heuristic_prior,
            correction_logits,
        )
        losses = self.compute_router_losses(heuristic_prior, calibrated_policy)

        diagnostics["correction_logits"] = dict(correction_logits)
        diagnostics["calibrated_policy"] = dict(calibrated_policy)

        return RouterState(
            heuristic_prior=heuristic_prior,
            calibrated_policy=calibrated_policy,
            diagnostics=diagnostics,
            losses=losses,
        )


# === EXPERT MANAGER ===

class ExpertManager:
    """
    Run the four pretrained quadrant experts in dense mode.

    Responsibilities:
    - load expert modules/checkpoints
    - execute each expert on the same base representation
    - return outputs in a common structure for editor-side fusion

    Important:
    - this component does not decide expert weights
    - all four experts should be available to the editor in dense mode
    - expert identities and iteration order follow CANONICAL_QUADRANT_ORDER
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        expert_config: ExpertConfig,
        generation_config: GenerationConfig,
    ) -> None:
        # store shared model/tokenizer references and load expert checkpoints
        raise NotImplementedError

    def load_experts(self) -> None:
        """
        Load all four quadrant experts without merging them into the base model.

        Experts (keys follow CANONICAL_QUADRANT_ORDER):
        - left_lib
        - left_auth
        - right_lib
        - right_auth
        """
        raise NotImplementedError

    def run_single_expert(
        self,
        expert_name: str,
        prompt_text: str,
        prompt_state: PromptState,
    ) -> ExpertOutput:
        """
        Execute one expert on the current prompt.

        Returns:
        - hidden-state output for editor-side fusion
        - optional decoded candidate text for logging or fallback synthesis
        """
        raise NotImplementedError

    def run_all_experts(
        self,
        prompt_text: str,
        prompt_state: PromptState,
    ) -> dict[str, ExpertOutput]:
        """
        Run all experts in dense mode.

        Logic:
        - preserve per-expert outputs for recursive editing
        - return a shared structure suitable for aggregation and trace logging
        """
        raise NotImplementedError


# === EDITOR ===

class Editor:
    """
    Recursively fuse expert hidden states into a debiased mixed hidden state.

    Logic:
    - initialize mixture weights from the router policy (or uniform, per
      EditorConfig.initialization_mode)
    - aggregate expert hidden states into a fused state
    - score current ideological alignment of the mixture
    - compute correction based on current alignment
    - update weights and re-aggregate
    - stop after convergence or max_edit_steps

    Inputs:
    - consumes RouterState as produced by Router.route()
    - mixture weights are keyed by CANONICAL_QUADRANT_ORDER, matching router
      output and ExpertConfig / ExpertManager naming

    Output ownership:
    - the editor returns an EditorResult: final mixed hidden state, final
      alpha and alignment, per-step traces, and run-level metadata
    - the editor does NOT decode text; downstream decoding into the final
      generated answer is owned by MoCEEngine.run
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        input_transformer: InputTransformer,
        config: EditorConfig,
        generation_config: GenerationConfig,
    ) -> None:
        # keep access to the base model, tokenizer, projector, and editor hyperparameters
        raise NotImplementedError

    def _validate_canonical_mapping(self, mapping: Any, name: str) -> None:
        """
        Validate that a mapping is a dict keyed exactly by CANONICAL_QUADRANT_ORDER.

        Logic:
        - dict-ness only; value types and value semantics are checked by callers
        - missing or extra keys raise ValueError naming them
        - never relies on dict insertion order

        Raises:
        - ValueError if mapping is not a dict, or its key set does not equal
          CANONICAL_QUADRANT_ORDER exactly
        """
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{name} must be a dict, got {type(mapping).__name__}"
            )
        expected_keys = set(CANONICAL_QUADRANT_ORDER)
        actual_keys = set(mapping.keys())
        missing_keys = expected_keys - actual_keys
        if missing_keys:
            raise ValueError(
                f"{name} is missing required keys: {sorted(missing_keys)}; "
                f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
            )
        unexpected_keys = actual_keys - expected_keys
        if unexpected_keys:
            raise ValueError(
                f"{name} has unexpected keys: {sorted(unexpected_keys)}; "
                f"expected exactly {list(CANONICAL_QUADRANT_ORDER)}"
            )

    def _ordered_quadrant_values(
        self,
        mapping: dict[str, float],
        name: str,
    ) -> list[float]:
        """
        Return values from a canonical-key mapping in CANONICAL_QUADRANT_ORDER.

        Logic:
        - validate keys via _validate_canonical_mapping
        - read values in canonical order; never depend on dict insertion order
        - cast each value to float for downstream numeric use

        Raises:
        - ValueError if the mapping fails canonical-key validation
        """
        self._validate_canonical_mapping(mapping, name)
        return [float(mapping[key]) for key in CANONICAL_QUADRANT_ORDER]

    def _mapping_from_ordered_values(
        self,
        values: list[float] | tuple[float, ...],
        name: str,
    ) -> dict[str, float]:
        """
        Convert a list/tuple of values aligned with CANONICAL_QUADRANT_ORDER
        into a canonically-keyed dict.

        Logic:
        - require exactly len(CANONICAL_QUADRANT_ORDER) entries (no padding,
          no truncation)
        - require numeric scalars; canonical key naming is preserved

        Raises:
        - ValueError if values is not a list/tuple, has the wrong length, or
          contains a non-numeric entry
        """
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"{name} must be a list or tuple, got {type(values).__name__}"
            )
        if len(values) != len(CANONICAL_QUADRANT_ORDER):
            raise ValueError(
                f"{name} must have exactly {len(CANONICAL_QUADRANT_ORDER)} entries "
                f"aligned with CANONICAL_QUADRANT_ORDER; got {len(values)}"
            )
        for index, value in enumerate(values):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name}[{index}] must be int or float, "
                    f"got {type(value).__name__}"
                )
        return {
            key: float(value)
            for key, value in zip(CANONICAL_QUADRANT_ORDER, values)
        }

    def _validate_policy_mapping(
        self,
        policy: Any,
        name: str,
        *,
        sum_tolerance: float = 1e-6,
    ) -> None:
        """
        Validate a probability distribution over CANONICAL_QUADRANT_ORDER.

        Requirements:
        - keys match CANONICAL_QUADRANT_ORDER exactly
        - values are int or float, finite (no NaN, no inf), strictly positive
        - values sum to 1 within sum_tolerance

        Raises:
        - ValueError naming the violated condition; no silent normalization
        """
        self._validate_canonical_mapping(policy, name)
        for key in CANONICAL_QUADRANT_ORDER:
            value = policy[key]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name}[{key!r}] must be int or float, "
                    f"got {type(value).__name__}"
                )
            if math.isnan(value):
                raise ValueError(f"{name}[{key!r}] is NaN")
            if math.isinf(value):
                raise ValueError(f"{name}[{key!r}] is infinite")
            if value <= 0:
                raise ValueError(
                    f"{name}[{key!r}] must be strictly positive for a policy; "
                    f"got {value}"
                )
        total = sum(float(policy[key]) for key in CANONICAL_QUADRANT_ORDER)
        if abs(total - 1.0) > sum_tolerance:
            raise ValueError(
                f"{name} must sum to 1 within {sum_tolerance}; got sum={total}"
            )

    def _validate_alignment_mapping(self, alignment: Any, name: str) -> None:
        """
        Validate a quadrant-alignment mapping over CANONICAL_QUADRANT_ORDER.

        Requirements:
        - keys match CANONICAL_QUADRANT_ORDER exactly
        - values are int or float and finite (no NaN, no inf)
        - no positivity or sum-to-1 constraint (alignment scores are signed
          and arbitrarily scaled, unlike policies)

        Raises:
        - ValueError naming the violated condition
        """
        self._validate_canonical_mapping(alignment, name)
        for key in CANONICAL_QUADRANT_ORDER:
            value = alignment[key]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name}[{key!r}] must be int or float, "
                    f"got {type(value).__name__}"
                )
            if math.isnan(value):
                raise ValueError(f"{name}[{key!r}] is NaN")
            if math.isinf(value):
                raise ValueError(f"{name}[{key!r}] is infinite")

    def _validate_expert_outputs(
        self,
        expert_outputs: dict[str, ExpertOutput],
    ) -> None:
        """
        Validate the dense expert-output dict consumed by the Editor.

        Requirements:
        - keys match CANONICAL_QUADRANT_ORDER exactly
        - each value is an ExpertOutput
        - each ExpertOutput.hidden_output is a non-None torch.Tensor with
          all entries finite (no NaN, no inf)
        - all hidden-state tensors share an identical shape

        Notes:
        - the Editor mixes ExpertOutput.hidden_output. The dataclass field is
          typed `Any | None = None` for upstream flexibility, but the Editor
          contract requires a real torch.Tensor here -- None is rejected
          loudly rather than silently broadening the contract.

        Raises:
        - ValueError naming the violated condition
        """
        self._validate_canonical_mapping(expert_outputs, "expert_outputs")

        reference_shape: tuple[int, ...] | None = None
        for key in CANONICAL_QUADRANT_ORDER:
            output = expert_outputs[key]
            if not isinstance(output, ExpertOutput):
                raise ValueError(
                    f"expert_outputs[{key!r}] must be an ExpertOutput, "
                    f"got {type(output).__name__}"
                )
            hidden_output = output.hidden_output
            if hidden_output is None:
                raise ValueError(
                    f"expert_outputs[{key!r}].hidden_output is None; "
                    "Editor mixing requires a torch.Tensor hidden state"
                )
            if not isinstance(hidden_output, torch.Tensor):
                raise ValueError(
                    f"expert_outputs[{key!r}].hidden_output must be a torch.Tensor, "
                    f"got {type(hidden_output).__name__}"
                )
            if not torch.isfinite(hidden_output).all().item():
                raise ValueError(
                    f"expert_outputs[{key!r}].hidden_output contains NaN or inf entries"
                )
            current_shape = tuple(hidden_output.shape)
            if reference_shape is None:
                reference_shape = current_shape
            elif current_shape != reference_shape:
                raise ValueError(
                    f"expert_outputs[{key!r}].hidden_output has shape {current_shape}; "
                    f"expected {reference_shape} (all expert hidden states must "
                    "share an identical shape)"
                )

    def _validate_prompt_state_for_editor(self, prompt_state: Any) -> None:
        """
        Validate the PromptState fields the Editor consumes.

        Requirements:
        - prompt_state is a PromptState instance
        - prompt_state.quadrant_scores is a valid alignment mapping over
          CANONICAL_QUADRANT_ORDER

        Notes:
        - this validation intentionally covers only what the Editor needs at
          this step (the alignment seed). Other PromptState fields (axis
          scores, hidden_representation, bias_magnitude) are validated by
          their own consumers and are not load-bearing for the Editor here.

        Raises:
        - ValueError naming the violated condition
        """
        if not isinstance(prompt_state, PromptState):
            raise ValueError(
                "prompt_state must be a PromptState, "
                f"got {type(prompt_state).__name__}"
            )
        self._validate_alignment_mapping(
            prompt_state.quadrant_scores,
            "prompt_state.quadrant_scores",
        )

    def _validate_router_state_for_editor(self, router_state: Any) -> None:
        """
        Validate the RouterState fields the Editor consumes.

        Requirements:
        - router_state is a RouterState instance
        - router_state.calibrated_policy is a valid policy mapping over
          CANONICAL_QUADRANT_ORDER

        Notes:
        - router-based initialization uses calibrated_policy. In heuristic-only
          mode Router.route() already mirrors heuristic_prior into
          calibrated_policy, so this single field is the agreed Editor entry
          point regardless of router mode.

        Raises:
        - ValueError naming the violated condition
        """
        if not isinstance(router_state, RouterState):
            raise ValueError(
                "router_state must be a RouterState, "
                f"got {type(router_state).__name__}"
            )
        self._validate_policy_mapping(
            router_state.calibrated_policy,
            "router_state.calibrated_policy",
        )

    def initialize_editor_weights(self, router_state: RouterState) -> dict[str, float]:
        """
        Initialize alpha (editor mixture weights) per EditorConfig.initialization_mode.

        Modes:
        - "router_policy": initialize alpha from RouterState.calibrated_policy.
          In heuristic router mode Router.route() already mirrors
          heuristic_prior into calibrated_policy, so this single field is the
          correct entry point regardless of router mode.
        - "uniform": initialize alpha as a uniform distribution
          (1 / len(CANONICAL_QUADRANT_ORDER)) over the canonical quadrants,
          ignoring router_state.

        Returns:
        - a fresh dict keyed by CANONICAL_QUADRANT_ORDER whose values form a
          validated probability distribution (strictly positive, finite,
          summing to 1 within policy-validator tolerance)

        Raises:
        - ValueError if EditorConfig.initialization_mode is not one of the
          supported values, or if the chosen source mapping fails policy
          validation
        """
        mode = self.config.initialization_mode
        if mode == "router_policy":
            self._validate_router_state_for_editor(router_state)
            return {
                key: float(router_state.calibrated_policy[key])
                for key in CANONICAL_QUADRANT_ORDER
            }
        if mode == "uniform":
            uniform_weight = 1.0 / len(CANONICAL_QUADRANT_ORDER)
            alpha = {key: uniform_weight for key in CANONICAL_QUADRANT_ORDER}
            self._validate_policy_mapping(alpha, "alpha (uniform initialization)")
            return alpha
        raise ValueError(
            "EditorConfig.initialization_mode must be one of "
            f"{{'router_policy', 'uniform'}}; got {mode!r}"
        )

    def _mix_hidden_states(
        self,
        expert_outputs: dict[str, ExpertOutput],
        alpha: dict[str, float],
    ) -> torch.Tensor:
        """
        Dense MoE fusion of expert hidden states under the current alpha.

        Rule:
        - mixed = sum_q alpha[q] * expert_outputs[q].hidden_output
          for q in CANONICAL_QUADRANT_ORDER

        Inputs:
        - expert_outputs: canonical-keyed mapping to ExpertOutput. Validated
          via _validate_expert_outputs, which already enforces the narrow
          Editor-side contract: each hidden_output is a non-None torch.Tensor
          with finite entries, and all four tensors share an identical shape.
        - alpha: probability distribution over CANONICAL_QUADRANT_ORDER.
          Validated via _validate_policy_mapping (canonical keys, finite,
          strictly positive, sums to 1).

        Computation:
        - iterate CANONICAL_QUADRANT_ORDER (no reliance on dict order)
        - seed the accumulator from the first expert (out-of-place
          multiplication) and add the remaining three with out-of-place
          arithmetic; expert tensors are never mutated in place
        - dtype/device: expert tensors are used as-is. _validate_expert_outputs
          guarantees identical shapes; dtype/device alignment is the caller's
          responsibility (typically guaranteed by a single base model and a
          single dense-mode forward pass over all four experts)

        Semantics:
        - this is dense MoE fusion across all four quadrant experts;
          no expert is dropped, skipped, or substituted with a fallback
        - the output remains in hidden-state space with the shared expert shape
        - this helper does not decode text and does not recompute alignment

        Returns:
        - a torch.Tensor in hidden-state space matching the shared shape
          of the input expert tensors

        Raises:
        - ValueError if expert_outputs fails expert-output validation, or
          alpha fails policy validation
        """
        self._validate_expert_outputs(expert_outputs)
        self._validate_policy_mapping(alpha, "alpha")

        first_key = CANONICAL_QUADRANT_ORDER[0]
        mixed = expert_outputs[first_key].hidden_output * float(alpha[first_key])
        for key in CANONICAL_QUADRANT_ORDER[1:]:
            mixed = mixed + expert_outputs[key].hidden_output * float(alpha[key])
        return mixed

    def aggregate_expert_outputs(
        self,
        expert_outputs: dict[str, ExpertOutput],
        weights: dict[str, float],
    ) -> Any:
        """
        Build fused hidden-state representation from dense expert outputs.

        Notes:
        - this is the first aggregation stage inside the editor
        - aggregation should remain interpretable and traceable across edit steps
        - the output stays in hidden-state space; no decoding happens here
        """
        raise NotImplementedError

    def score_current_mixture(
        self,
        mixed_hidden_state: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Recompute the political alignment of the current mixed hidden state.

        Inputs:
        - mixed_hidden_state: the fused hidden state produced by
          _mix_hidden_states. Must be a torch.Tensor with finite entries.

        Output (fresh dict, plain Python floats):
        - "economic_score": float
        - "social_score":   float
        - "quadrant_scores": dict[str, float]   # rebuilt in CANONICAL_QUADRANT_ORDER
        - "bias_magnitude": float

        Scoring path:
        - mirrors InputTransformer.transform on a hidden state:
          maybe-center the representation, compute axis scores,
          compute quadrant scores, derive bias magnitude
        - reuses the Editor's stored InputTransformer (self.input_transformer);
          the Editor does not load steering vectors itself

        Validation:
        - mixed_hidden_state is a torch.Tensor with all-finite entries
        - axis-score output contains numeric, finite economic_score and
          social_score
        - quadrant_scores passes _validate_alignment_mapping (canonical
          keys, numeric, finite); rebuilt in canonical order so the
          projection helper's dict insertion order is never trusted
        - bias_magnitude is numeric, finite, non-negative

        Semantics:
        - this helper recomputes the alignment of the fused hidden state
          for use by recursive editing, step traces, and final diagnostics
        - it does not update alpha, mix experts, or decode text
        - steering-vector loading is delegated to InputTransformer

        Raises:
        - ValueError naming the violated condition; no silent coercion of
          InputTransformer outputs
        """
        if not isinstance(mixed_hidden_state, torch.Tensor):
            raise ValueError(
                "mixed_hidden_state must be a torch.Tensor, "
                f"got {type(mixed_hidden_state).__name__}"
            )
        if not torch.isfinite(mixed_hidden_state).all().item():
            raise ValueError("mixed_hidden_state contains NaN or inf entries")

        # mirror the InputTransformer pipeline used to build PromptState:
        # encode is skipped (we already hold a hidden state), then
        # maybe_center -> axis_scores -> quadrant_scores -> bias_magnitude
        centered = self.input_transformer.maybe_center_representation(mixed_hidden_state)
        axis_scores = self.input_transformer.compute_axis_scores(centered)
        quadrant_scores = self.input_transformer.compute_quadrant_scores(centered)

        if not isinstance(axis_scores, dict):
            raise ValueError(
                "InputTransformer.compute_axis_scores must return a dict, "
                f"got {type(axis_scores).__name__}"
            )
        for required_key in ("economic_score", "social_score"):
            if required_key not in axis_scores:
                raise ValueError(
                    f"InputTransformer.compute_axis_scores output is missing "
                    f"required key {required_key!r}"
                )

        economic_score = axis_scores["economic_score"]
        social_score = axis_scores["social_score"]
        for name, value in (
            ("economic_score", economic_score),
            ("social_score", social_score),
        ):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"InputTransformer.compute_axis_scores output {name!r} "
                    f"must be int or float, got {type(value).__name__}"
                )
            if math.isnan(value):
                raise ValueError(
                    f"InputTransformer.compute_axis_scores output {name!r} is NaN"
                )
            if math.isinf(value):
                raise ValueError(
                    f"InputTransformer.compute_axis_scores output {name!r} is infinite"
                )

        self._validate_alignment_mapping(
            quadrant_scores,
            "InputTransformer.compute_quadrant_scores",
        )

        bias_magnitude = self.input_transformer.compute_bias_magnitude(
            float(economic_score),
            float(social_score),
        )
        if not isinstance(bias_magnitude, (int, float)):
            raise ValueError(
                "InputTransformer.compute_bias_magnitude must return int or float, "
                f"got {type(bias_magnitude).__name__}"
            )
        if math.isnan(bias_magnitude):
            raise ValueError("InputTransformer.compute_bias_magnitude returned NaN")
        if math.isinf(bias_magnitude):
            raise ValueError("InputTransformer.compute_bias_magnitude returned infinity")
        if bias_magnitude < 0:
            raise ValueError(
                "InputTransformer.compute_bias_magnitude must be non-negative; "
                f"got {bias_magnitude}"
            )

        return {
            "economic_score": float(economic_score),
            "social_score": float(social_score),
            "quadrant_scores": {
                key: float(quadrant_scores[key]) for key in CANONICAL_QUADRANT_ORDER
            },
            "bias_magnitude": float(bias_magnitude),
        }

    def _compute_delta_from_alignment(
        self,
        alignment: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute the additive log-space correction term delta from a quadrant
        alignment mapping.

        Rule (v1):
        - delta[q] = -self.config.correction_beta * alignment[q]
          for each q in CANONICAL_QUADRANT_ORDER

        Semantics:
        - larger positive alignment for a quadrant yields a more negative
          delta for that quadrant; under the later softmax update
          alpha_next = softmax(log(alpha) + delta), more negatively
          aligned quadrants get relatively boosted toward neutral
        - this helper computes only the additive correction term;
          it does not normalize weights, does not apply softmax, and
          does not consume the current alpha

        Inputs:
        - alignment: canonical quadrant-score mapping (signed real
          scalars over CANONICAL_QUADRANT_ORDER); validated via
          _validate_alignment_mapping before use

        Returns:
        - a fresh dict keyed by CANONICAL_QUADRANT_ORDER whose values
          are plain Python floats

        Raises:
        - ValueError if alignment fails alignment-mapping validation
          (key set, numeric type, finiteness)
        """
        self._validate_alignment_mapping(alignment, "alignment")
        beta = self.config.correction_beta
        return {
            key: float(-beta * alignment[key])
            for key in CANONICAL_QUADRANT_ORDER
        }

    def compute_editor_correction(
        self,
        prompt_state: PromptState,
        current_alignment: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute correction signal for editor-side weight updates.

        Logic:
        - penalize experts aligned with the current ideological drift
        - boost counterbalancing experts that pull the mixture toward center
        - keep the correction geometry aligned with the original debias plan
        """
        raise NotImplementedError

    def _softmax(self, logits: list[float]) -> list[float]:
        """
        Numerically stable softmax over a list of finite logits.

        Logic:
        - validate the list (non-empty, numeric, finite per entry)
        - subtract max(logits) before exponentiation for numerical stability
        - normalize exponentials by their sum

        Notes:
        - mirrors Router._softmax style; kept local so Editor numeric
          logic is self-contained without cross-class references

        Raises:
        - ValueError if the logits list is empty or contains a non-numeric,
          NaN, or inf entry
        """
        if len(logits) == 0:
            raise ValueError("_softmax received an empty logits list")
        for index, value in enumerate(logits):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"_softmax logits[{index}] must be int or float, "
                    f"got {type(value).__name__}"
                )
            if math.isnan(value):
                raise ValueError(f"_softmax logits[{index}] is NaN")
            if math.isinf(value):
                raise ValueError(f"_softmax logits[{index}] is infinite")

        max_logit = max(logits)
        shifted_exps = [math.exp(value - max_logit) for value in logits]
        total = sum(shifted_exps)
        return [exp_value / total for exp_value in shifted_exps]

    def _update_alpha(
        self,
        alpha_current: dict[str, float],
        delta: dict[str, float],
    ) -> dict[str, float]:
        """
        Apply the additive log-space correction to alpha.

        Rule:
        - alpha_next = softmax(log(alpha_current) + delta)

        Inputs:
        - alpha_current: probability distribution over CANONICAL_QUADRANT_ORDER.
          Validated as a policy mapping (canonical keys, numeric, finite,
          strictly positive, sums to 1). Strict positivity makes log() safe
          without epsilon smoothing.
        - delta: signed real-valued additive correction over the same
          canonical key set. Validated as an alignment-style mapping
          (canonical keys, numeric, finite). delta is NOT itself a
          probability distribution; softmax is the only normalization.

        Computation:
        - iterate CANONICAL_QUADRANT_ORDER (no reliance on dict order)
        - combined_logit_i = log(alpha_current_i) + delta_i
        - alpha_next = self._softmax(combined_logits)

        Properties:
        - zero delta yields alpha_next == alpha_current up to floating-point
          tolerance
        - increasing delta_i monotonically increases alpha_next_i
        - alpha_next is strictly positive and sums to 1 (softmax invariants)

        Returns:
        - a fresh dict keyed by CANONICAL_QUADRANT_ORDER mapping to plain
          Python floats; these are the updated Editor mixture weights,
          not expert outputs

        Raises:
        - ValueError if alpha_current fails policy validation, or delta
          fails alignment-mapping validation
        """
        self._validate_policy_mapping(alpha_current, "alpha_current")
        self._validate_alignment_mapping(delta, "delta")
        combined_logits = [
            math.log(float(alpha_current[key])) + float(delta[key])
            for key in CANONICAL_QUADRANT_ORDER
        ]
        next_values = self._softmax(combined_logits)
        return {
            key: prob
            for key, prob in zip(CANONICAL_QUADRANT_ORDER, next_values)
        }

    def update_editor_weights(
        self,
        current_weights: dict[str, float],
        correction_signal: dict[str, float],
    ) -> dict[str, float]:
        """
        Update mixture weights using correction-adjusted softmax.

        Logic:
        - compute alpha = softmax(log(alpha_0) + Delta)
        - maintain normalized expert contributions at every edit step
        """
        raise NotImplementedError

    def should_stop(
        self,
        previous_weights: dict[str, float],
        updated_weights: dict[str, float],
        step_index: int,
    ) -> bool:
        """
        Decide whether recursive editing should stop.

        Stopping criteria may include:
        - step limit reached
        - small weight change
        - negligible improvement in mixture alignment
        """
        raise NotImplementedError

    def _run_edit_loop(
        self,
        initial_alpha: dict[str, float],
        initial_alignment: dict[str, float],
        expert_outputs: dict[str, ExpertOutput],
    ) -> EditorResult:
        """
        Reusable Editor step loop.

        Inputs:
        - initial_alpha: starting mixture weights, validated as a policy
          mapping over CANONICAL_QUADRANT_ORDER
        - initial_alignment: starting alignment (quadrant_scores only),
          validated as an alignment mapping over CANONICAL_QUADRANT_ORDER
        - expert_outputs: dense expert hidden states, validated via
          _validate_expert_outputs

        Step semantics (per iteration):
        1. delta = self._compute_delta_from_alignment(current_alignment)
        2. alpha_next = self._update_alpha(current_alpha, delta)
        3. mixed_hidden_state = self._mix_hidden_states(expert_outputs, alpha_next)
        4. mixture_scores = self.score_current_mixture(mixed_hidden_state)
        5. alignment_next = mixture_scores["quadrant_scores"]
        6. max_alpha_change = max_q |alpha_next[q] - current_alpha[q]|
           max_alignment_change = max_q |alignment_next[q] - current_alignment[q]|
        7. if EditorConfig.keep_edit_trace, append an EditorStepTrace with
           copies of every per-step dict (no aliases)

        Stopping rules:
        - use_recursive_editing=False: run exactly 1 step (max_edit_steps
          ignored). Terminal state has stopped_early=False, stop_reason=None.
        - use_recursive_editing=True: run up to max_edit_steps. After each
          step, stop early if both max_alpha_change <= convergence_threshold
          and max_alignment_change <= convergence_threshold, with
          stopped_early=True and stop_reason="converged". Exhausting the
          budget without convergence yields stopped_early=False,
          stop_reason=None.

        Returns:
        - EditorResult populated with final_mixed_hidden_state, final_alpha
          (fresh copy), final_alignment (quadrant_scores from the last
          mixture_scores; fresh copy), step_traces (possibly empty),
          num_steps_run, stopped_early, stop_reason

        Raises:
        - ValueError on malformed inputs or on
          EditorConfig.max_edit_steps that is not a positive int
        """
        self._validate_policy_mapping(initial_alpha, "initial_alpha")
        self._validate_alignment_mapping(initial_alignment, "initial_alignment")
        self._validate_expert_outputs(expert_outputs)

        if self.config.use_recursive_editing:
            max_steps = self.config.max_edit_steps
        else:
            max_steps = 1
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise ValueError(
                "EditorConfig.max_edit_steps must be a positive int; "
                f"got {self.config.max_edit_steps!r}"
            )

        current_alpha = dict(initial_alpha)
        current_alignment = dict(initial_alignment)
        step_traces: list[EditorStepTrace] = []
        stopped_early = False
        stop_reason: str | None = None
        num_steps_run = 0

        for step_index in range(max_steps):
            delta = self._compute_delta_from_alignment(current_alignment)
            alpha_next = self._update_alpha(current_alpha, delta)
            mixed_hidden_state = self._mix_hidden_states(expert_outputs, alpha_next)
            mixture_scores = self.score_current_mixture(mixed_hidden_state)
            alignment_next = mixture_scores["quadrant_scores"]

            max_alpha_change = max(
                abs(alpha_next[q] - current_alpha[q])
                for q in CANONICAL_QUADRANT_ORDER
            )
            max_alignment_change = max(
                abs(alignment_next[q] - current_alignment[q])
                for q in CANONICAL_QUADRANT_ORDER
            )

            if self.config.keep_edit_trace:
                step_traces.append(
                    EditorStepTrace(
                        step_index=step_index,
                        alpha_before=dict(current_alpha),
                        delta=dict(delta),
                        alpha_after=dict(alpha_next),
                        alignment_before=dict(current_alignment),
                        alignment_after=dict(alignment_next),
                        max_alpha_change=float(max_alpha_change),
                        max_alignment_change=float(max_alignment_change),
                    )
                )

            current_alpha = alpha_next
            current_alignment = alignment_next
            num_steps_run = step_index + 1

            if self.config.use_recursive_editing and (
                max_alpha_change <= self.config.convergence_threshold
                and max_alignment_change <= self.config.convergence_threshold
            ):
                stopped_early = True
                stop_reason = "converged"
                break

        # max_steps >= 1 is enforced above, so the loop ran at least once and
        # mixed_hidden_state / current_alpha / current_alignment are bound.
        return EditorResult(
            final_mixed_hidden_state=mixed_hidden_state,
            final_alpha=dict(current_alpha),
            final_alignment=dict(current_alignment),
            step_traces=step_traces,
            num_steps_run=num_steps_run,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )

    def run_editing_loop(
        self,
        prompt_text: str,
        prompt_state: PromptState,
        router_state: RouterState,
        expert_outputs: dict[str, ExpertOutput],
    ) -> EditorResult:
        """
        Full recursive editor loop.

        Flow:
        - initialize editor weights (per EditorConfig.initialization_mode)
        - build initial fused hidden state
        - score current mixture alignment
        - compute correction signal
        - update weights and re-aggregate
        - repeat until stable or max_edit_steps reached
        - package the final mixed hidden state, final alpha, final alignment,
          per-step traces, and run-level metadata into an EditorResult

        Notes:
        - keep recursion shallow in v1; one-step update is the default
        - retain full step traces for interpretability and downstream evaluation
        - decoding the final hidden state into text is owned by MoCEEngine.run,
          not by the editor
        """
        raise NotImplementedError


# === ENGINE ===

class MoCEEngine:
    """
    Main reusable debiasing engine.

    Flow:
    - transform prompt
    - compute routing prior/policy
    - run all experts
    - recursively edit and fuse outputs
    - return final text and full trace

    Important:
    - keep this class architecture-only
    - do not add experiment loops, benchmark logic, or output-directory management here
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        steering_config: SteeringVectorConfig,
        router_config: RouterConfig,
        expert_config: ExpertConfig,
        editor_config: EditorConfig,
        generation_config: GenerationConfig,
    ) -> None:
        # instantiate all reusable architecture components
        # InputTransformer handles political-state extraction
        # Router builds pi_0 and optional calibrated policy pi
        # ExpertManager runs the four quadrant specialists
        # Editor recursively fuses expert outputs into the final answer
        self.router = Router(router_config)

    def run(self, prompt_text: str) -> MoCEResult:
        """
        Execute the full debiasing pipeline for a single prompt.

        Pipeline:
        1. transform prompt into compass-space diagnostics
        2. compute heuristic routing prior pi_0
        3. optionally calibrate router policy pi around pi_0
        4. run all four quadrant experts in dense mode
        5. recursively fuse expert hidden states through the editor; this
           returns an EditorResult (final mixed hidden state plus metadata)
        6. decode the final mixed hidden state into the final answer
        7. package prompt/router/expert/editor intermediates and the decoded
           text into a MoCEResult
        """
        raise NotImplementedError
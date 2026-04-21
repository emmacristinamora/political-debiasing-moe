# src/06_moce_architecture.py


# === IMPORTS ===

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple


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

    Notes:
    - v1 may run in heuristic-only mode
    - the interface should still expose calibrated mode so teammates can extend it later
    """

    use_calibrated_router: bool = False
    beta: float = 1.0
    temperature: float = 1.0
    kl_weight: float = 0.1
    entropy_weight: float = 0.01
    router_hidden_dim: int = 128
    fallback_to_uniform_if_centered: bool = True
    center_threshold: float = 0.05


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
    - aggregate expert outputs into a fused representation
    - compute correction from ideological alignment of the current mixture
    - update weights and recompute until convergence or max steps

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

    Notes:
    - in heuristic-only mode, pi may equal pi_0
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
    - current weights before and after correction
    - correction signal used at this step
    - ideological score of the current fused mixture
    - optional intermediate decoded text
    """

    step_index: int
    input_weights: dict[str, float]
    correction_signal: dict[str, float]
    updated_weights: dict[str, float]
    mixture_alignment: dict[str, float]
    intermediate_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MoCEResult:
    """
    Full output of one MoCE run.

    This should be rich enough that 07_run_moce.py only needs to save it,
    not reconstruct anything after the fact.
    """

    prompt_text: str
    prompt_state: PromptState
    router_state: RouterState
    expert_outputs: dict[str, ExpertOutput]
    editor_trace: list[EditorStepTrace]
    final_weights: dict[str, float]
    final_text: str
    final_alignment: dict[str, float]
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


# === ROUTER ===

class Router:
    """
    Compute initial expert routing for debiasing.

    Logic:
    - build a heuristic prior pi_0 from quadrant alignment scores
    - optionally apply a lightweight calibrated correction around log(pi_0)
    - expose KL and entropy terms for router training

    Important:
    - prompts near a quadrant should downweight that quadrant and upweight
      opposite or adjacent quadrant
    - the calibrated router should not learn a free policy from scratch;
      it should learn a small correction around the heuristic prior
    """

    def __init__(self, config: RouterConfig) -> None:
        # store router hyperparameters and initialize optional calibration module
        raise NotImplementedError

    def build_heuristic_prior(self, prompt_state: PromptState) -> dict[str, float]:
        """
        Build heuristic prior pi_0 from quadrant alignment scores.

        Logic:
        - use counterbalancing geometry rather than same-direction amplification
        - high alignment with one quadrant should penalize that quadrant in pi_0
        - prompts near the center may optionally fall back to a uniform prior
        """
        raise NotImplementedError

    def compute_router_correction(self, prompt_state: PromptState) -> dict[str, float]:
        """
        Compute lightweight calibrated correction around log(pi_0).

        Notes:
        - in heuristic-only mode this can return zeros or be skipped
        - this is the delta(h) term from the original routing plan
        """
        raise NotImplementedError

    def combine_prior_and_correction(
        self,
        heuristic_prior: dict[str, float],
        correction_logits: dict[str, float],
    ) -> dict[str, float]:
        """
        Combine heuristic prior and correction into calibrated policy pi.

        Logic:
        - compute pi = softmax(log(pi_0) + delta(h))
        - keep routing behavior close to the intended debiasing geometry
        """
        raise NotImplementedError

    def compute_router_losses(
        self,
        heuristic_prior: dict[str, float],
        calibrated_policy: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute router regularization losses.

        Includes:
        - KL(pi || pi_0) to anchor learned routing to the debias prior
        - entropy regularization to prevent collapse or overconfidence

        Notes:
        - these losses may be unused in inference-only mode, but the interface
          should still expose them for future router training
        """
        raise NotImplementedError

    def route(self, prompt_state: PromptState) -> RouterState:
        """
        Full routing pipeline.

        Flow:
        - build heuristic prior pi_0
        - optionally compute calibrated policy pi around pi_0
        - compute diagnostics and optional losses
        - return RouterState for downstream editing
        """
        raise NotImplementedError


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

        Experts:
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
    Recursively fuse expert outputs into a more politically neutral final answer.

    Logic:
    - initialize mixture weights from the router
    - aggregate expert outputs into a fused state
    - compute correction based on current ideological alignment
    - update weights and recompute the mixture
    - stop after convergence or max_edit_steps

    Important:
    - the editor owns finalization
    - final output is produced by the editor, not by a separate output transformer
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

    def initialize_editor_weights(self, router_state: RouterState) -> dict[str, float]:
        """
        Initialize editor weights from the router policy unless overridden.

        Logic:
        - default to calibrated router policy if available
        - fall back to heuristic prior or uniform weights if needed
        """
        raise NotImplementedError

    def aggregate_expert_outputs(
        self,
        expert_outputs: dict[str, ExpertOutput],
        weights: dict[str, float],
    ) -> Any:
        """
        Build fused representation from dense expert outputs.

        Notes:
        - this is the first aggregation stage inside the editor
        - aggregation should remain interpretable and traceable across edit steps
        """
        raise NotImplementedError

    def decode_fused_representation(
        self,
        fused_representation: Any,
        prompt_text: str,
    ) -> str:
        """
        Decode current fused state into text.

        Important:
        - decoding is part of editor finalization, not a separate component
        """
        raise NotImplementedError

    def score_current_mixture(
        self,
        fused_representation: Any,
        decoded_text: str | None = None,
    ) -> dict[str, float]:
        """
        Recompute ideological alignment of the current mixture.

        Logic:
        - use current mixture alignment, not only original prompt alignment
        - this enables recursive correction rather than one-shot prompt-based editing
        """
        raise NotImplementedError

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

    def run_editing_loop(
        self,
        prompt_text: str,
        prompt_state: PromptState,
        router_state: RouterState,
        expert_outputs: dict[str, ExpertOutput],
    ) -> tuple[str, dict[str, float], dict[str, float], list[EditorStepTrace]]:
        """
        Full recursive editor loop.

        Flow:
        - initialize editor weights
        - build initial fused representation
        - decode and rescore current mixture
        - compute correction signal
        - update weights and re-aggregate
        - repeat until stable
        - return final text, final weights, final alignment, and edit trace

        Notes:
        - keep recursion shallow in v1; one-step update is the default
        - retain full trace for interpretability and downstream evaluation
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
        raise NotImplementedError

    def run(self, prompt_text: str) -> MoCEResult:
        """
        Execute the full debiasing pipeline for a single prompt.

        Pipeline:
        1. transform prompt into compass-space diagnostics
        2. compute heuristic routing prior pi_0
        3. optionally calibrate router policy pi around pi_0
        4. run all four quadrant experts in dense mode
        5. recursively fuse expert outputs through the editor
        6. return final answer together with routing/editor traces
        """
        raise NotImplementedError
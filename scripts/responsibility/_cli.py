"""Command-line flags shared by the responsibility scripts, so that every
stage builds its ResponsibilityConfig the same way."""

import argparse

from src.responsibility.interaction import QUERY_MODES, SELECTION_MODES, InteractionConfig
from src.responsibility.metrics import COURTESY_ESTIMATORS, DIRECTIONS, OTHERS_FUTURES, ResponsibilityConfig
from src.responsibility.motion_set import NEIGHBOR_FUTURE_MODES


def _none_if_nonpositive(value):
    return value if value is not None and value > 0 else None


def add_model_args(p: argparse.ArgumentParser, ckpt_required: bool = True) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--ckpt", required=ckpt_required, help="SMART .ckpt checkpoint.")
    g.add_argument("--model-config", default="smart", help="configs/model/<name>.yaml group.")
    g.add_argument("--device", default="auto", help="cuda, cpu, or auto (cuda if available).")
    g.add_argument("--rollout-batch-size", type=int, default=0,
                   help="Scene copies per model call; 0 = auto (128 on GPU, 8 on CPU). "
                        "Tune with scripts/responsibility/benchmark_rollouts.py.")
    g.add_argument("--motion-set-num-k", type=int, default=2048,
                   help="Top-k of the sampler; 2048 = the whole vocabulary, i.e. exact sampling "
                        "(required by --courtesy-estimator exact, as is --motion-set-temp 1).")
    g.add_argument("--motion-set-temp", type=float, default=1.0)


def add_responsibility_args(p: argparse.ArgumentParser) -> None:
    d = ResponsibilityConfig()
    i = InteractionConfig()

    g = p.add_argument_group("responsibility (paper Sec. III)")
    g.add_argument("--n-safety-samples", type=int, default=d.n_safety_samples)
    g.add_argument("--cvar-alpha", type=float, default=d.cvar_alpha)
    g.add_argument("--n-courtesy-samples", type=int, default=d.n_courtesy_samples)
    g.add_argument("--d-sat", type=float, default=d.d_sat, help="D_g saturation (m); <=0 disables.")
    g.add_argument("--metric-horizon", type=int, default=d.metric_horizon,
                   help="Future steps @10Hz over which beta is measured.")
    g.add_argument("--window-stride", type=int, default=d.window_stride,
                   help="Context steps between responsibility windows within a clip.")
    g.add_argument("--courtesy-estimator", default=d.courtesy_estimator, choices=COURTESY_ESTIMATORS,
                   help="boltzmann: the paper's Sec. V-B proxy (Eq. 7), zero unless the target's sampled "
                        "motions collide with the queried agent; exact: the model's own KL (Eq. 4) "
                        "via the chain rule over replayed samples.")
    g.add_argument("--courtesy-skip-distance", type=float, default=d.courtesy_skip_distance,
                   help="Skip courtesy rollouts for pairs further apart (m); <=0 or unset disables "
                        "(the default; 20 is exact for the boltzmann estimator, lossy for exact).")
    g.add_argument("--others-future", default=d.others_future, choices=OTHERS_FUTURES,
                   help="Future of the other agents when featurising trajectories for the reward "
                        "networks: constant_velocity (causal; use for prediction) or logged "
                        "(offline analysis only: leaks the answer into prediction).")
    g.add_argument("--plausible-fraction", type=float, default=d.plausible_fraction,
                   help="Take the safety CVaR over only this most-probable fraction of the "
                        "queried agent's samples (1.0 keeps all, the paper's definition).")
    g.add_argument("--route-consistent", type=float, default=0.0, metavar="METRES",
                   help="Take the safety CVaR over only the samples that stay within this distance of "
                        "the queried agent's logged route (speed free, route fixed); <=0 keeps all.")
    g.add_argument("--collision-weight", type=float, default=d.collision_weight)
    g.add_argument("--drivable-half-width", type=float, default=d.drivable_half_width)
    g.add_argument("--neighbor-future", default=d.neighbor_future, choices=NEIGHBOR_FUTURE_MODES,
                   help="What the queried agent may foresee of its neighbours: logged (their "
                        "whole logged future), hidden (only up to now; the paper's open-loop "
                        "reading), simulated (nobody teacher-forced).")

    g = p.add_argument_group("who is queried, and against whom")
    g.add_argument("--query", default=d.query_mode, choices=QUERY_MODES,
                   help="vehicles, sdc (the self-driving car), interest (WOMD objects of "
                        "interest / tracks to predict), adv (the objects of interest other than "
                        "the SDC) or all.")
    g.add_argument("--direction", default=d.query_direction, choices=DIRECTIONS,
                   help="With --query sdc: outgoing (SDC toward others), incoming (others "
                        "toward the SDC) or both.")
    g.add_argument("--agent-type", type=int, default=0, help="With --query vehicles: 0=veh, 1=ped, 2=cyc, -1=all.")
    g.add_argument("--neighbor-selection", default=i.mode, choices=SELECTION_MODES,
                   help="radius (everyone within --neighbor-radius now), geometric (gap / PET / "
                        "TTC over the metric horizon) or influence (geometric plus model influence).")
    g.add_argument("--neighbor-radius", type=float, default=d.neighbor_radius, help="radius mode only.")
    g.add_argument("--prefilter-radius", type=float, default=i.prefilter_radius)
    g.add_argument("--gap-threshold", type=float, default=i.gap_threshold, help="m")
    g.add_argument("--pet-threshold", type=float, default=i.pet_threshold, help="s")
    g.add_argument("--ttc-threshold", type=float, default=i.ttc_threshold, help="s")
    g.add_argument("--influence-threshold", type=float, default=i.influence_threshold, help="nats per step")
    g.add_argument("--max-neighbors", type=int, default=i.max_neighbors, help="<=0 keeps all.")
    g.add_argument("--ooi-neighbours", action="store_true",
                   help="Only WOMD objects of interest can be neighbours (the selection rule above "
                        "still applies among them).")


def responsibility_config_from_args(args, num_historical_steps: int) -> ResponsibilityConfig:
    return ResponsibilityConfig(
        n_safety_samples=args.n_safety_samples,
        cvar_alpha=args.cvar_alpha,
        n_courtesy_samples=args.n_courtesy_samples,
        neighbor_radius=args.neighbor_radius,
        d_sat=_none_if_nonpositive(args.d_sat),
        metric_horizon=args.metric_horizon,
        window_stride=args.window_stride,
        num_historical_steps=num_historical_steps,
        neighbor_future=args.neighbor_future,
        collision_weight=args.collision_weight,
        drivable_half_width=args.drivable_half_width,
        courtesy_skip_distance=_none_if_nonpositive(args.courtesy_skip_distance),
        courtesy_estimator=args.courtesy_estimator,
        others_future=args.others_future,
        plausible_fraction=args.plausible_fraction,
        route_tolerance=_none_if_nonpositive(args.route_consistent),
        query_mode=args.query,
        query_direction=args.direction,
        interaction=InteractionConfig(
            mode=args.neighbor_selection,
            prefilter_radius=args.prefilter_radius,
            gap_threshold=args.gap_threshold,
            pet_threshold=args.pet_threshold,
            ttc_threshold=args.ttc_threshold,
            influence_threshold=args.influence_threshold,
            max_neighbors=_none_if_nonpositive(args.max_neighbors),
            ooi_only=args.ooi_neighbours,
        ),
    )


def agent_type_from_args(args):
    return None if args.agent_type < 0 else args.agent_type

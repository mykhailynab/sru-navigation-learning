#!/usr/bin/env python3
"""Compare tensorboard logs from multiple runs side by side.

Usage:
    # Compare all default runs
    python repro_scripts/compare_tb_logs.py

    # Compare specific runs by name
    python repro_scripts/compare_tb_logs.py --runs base lstm

    # Compare custom paths
    python repro_scripts/compare_tb_logs.py --extra "my_run:/path/to/events"

    # Save to file
    python repro_scripts/compare_tb_logs.py --save comparison.png
"""

import argparse
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_RUNS = {
    "SRU+DML": "logs/rsl_rl/b2w_navigation_mdpo/2026-05-22_11-55-51/events.out.tfevents.1779465360.guacamole.1548052.0",
    "LSTM+DML": "logs/rsl_rl/b2w_navigation_mdpo_lstm/2026-06-09_11-51-53/events.out.tfevents.1781020315.guacamole.1764439.0",
    "Ablate Proprioceptive": "logs/rsl_rl/b2w_navigation_mdpo/2026-06-08_18-26-06/events.out.tfevents.1780957568.guacamole.1568647.0",
    "Ball Target": "logs/rsl_rl/b2w_navigation_mdpo_ball/2026-06-10_18-26-24/events.out.tfevents.1781130386.guacamole.2087023.0",
}


PLOT_GROUPS = [
    {"title": "Mean Episode Length", "filter": "Train/mean_episode_length"},
    {"title": "Mean Reward", "filter": "Train/mean_reward"},
    {"title": "Robot Goal", "filter": "Metrics/robot_goal/"},
    {"title": "Value Function", "filter": "Loss/value_function"},
    {"title": "KL Divergence", "filter": "Loss/kl_divergence"},
    {"title": "Reach Goal XY Tight", "filter": "Episode_Reward/reach_goal_xy_tight"},
    {"title": "Episode Termination", "filter": "Episode_Termination/time_out"},
    {"title": "Success Rate", "filter": "Metrics/robot_goal/success_rate"},
]

COLS = 3

# Line styles to distinguish runs
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def load_events(path, label):
    ea = EventAccumulator(path)
    ea.Reload()
    tags = sorted(ea.Tags()["scalars"])
    if not tags:
        print(f"No scalar data found for {label}: {path}")
        sys.exit(1)
    return ea, tags


def find_matching_tags(tags, filt):
    return [t for t in tags if t.lower().startswith(filt.lower())]


def plot_comparison(runs, save_path=None):
    """Plot comparison of multiple runs.

    Args:
        runs: list of (name, EventAccumulator, tags) tuples
        save_path: optional path to save the figure
    """
    import matplotlib.pyplot as plt

    rows = (len(PLOT_GROUPS) + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(7 * COLS, 3.5 * rows), squeeze=False)

    for idx, group in enumerate(PLOT_GROUPS):
        ax = axes[idx // COLS][idx % COLS]
        filt = group["filter"]

        # Collect all matching subtags across runs
        all_subtags = set()
        run_matched = []
        for name, ea, tags in runs:
            matched = find_matching_tags(tags, filt)
            run_matched.append((name, ea, matched))
            all_subtags.update(matched)
        all_subtags = sorted(all_subtags)

        if not all_subtags:
            ax.set_title(f"{group['title']} (no data)", fontsize=11)
            ax.set_visible(False)
            continue

        for tag in all_subtags:
            short = tag.split("/")[-1] if "/" in tag else tag

            for i, (name, ea, matched) in enumerate(run_matched):
                if tag not in matched:
                    continue
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                ls = LINE_STYLES[i % len(LINE_STYLES)]
                label = f"{name}: {short}" if len(all_subtags) > 1 else name
                ax.plot(steps, values, label=label, linewidth=1.2, linestyle=ls)

        ax.set_title(group["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(PLOT_GROUPS), rows * COLS):
        axes[idx // COLS][idx % COLS].set_visible(False)

    title = " vs ".join(name for name, _, _ in runs)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare tensorboard logs from multiple runs.")
    parser.add_argument(
        "--runs", type=str, nargs="*", default=None,
        help=f"Names of default runs to compare. Available: {list(DEFAULT_RUNS.keys())}. "
             "If omitted, all default runs are used.",
    )
    parser.add_argument(
        "--extra", type=str, nargs="*", default=[],
        help='Additional runs as "name:path" pairs.',
    )
    parser.add_argument("--save", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    # Build run list
    selected = {}
    if args.runs is not None:
        for name in args.runs:
            if name not in DEFAULT_RUNS:
                print(f"Unknown run '{name}'. Available: {list(DEFAULT_RUNS.keys())}")
                sys.exit(1)
            selected[name] = DEFAULT_RUNS[name]
    else:
        selected = dict(DEFAULT_RUNS)

    for extra in args.extra:
        if ":" not in extra:
            print(f"Invalid --extra format '{extra}', expected 'name:path'")
            sys.exit(1)
        name, path = extra.split(":", 1)
        selected[name] = path

    if len(selected) < 2:
        print("Need at least 2 runs to compare.")
        sys.exit(1)

    # Load all runs
    loaded = []
    for name, path in selected.items():
        print(f"Loading {name}: {path}")
        ea, tags = load_events(path, name)
        print(f"  {len(tags)} tags found")
        loaded.append((name, ea, tags))

    plot_comparison(loaded, save_path=args.save)


if __name__ == "__main__":
    main()

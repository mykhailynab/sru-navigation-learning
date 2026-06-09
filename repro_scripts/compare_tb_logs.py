#!/usr/bin/env python3
"""Compare tensorboard logs from two runs side by side.

Usage:
    # Compare with defaults (base vs ablation)
    python repro_scripts/compare_tb_logs.py

    # Compare custom runs
    python repro_scripts/compare_tb_logs.py --base <path> --ablation <path>

    # Save to file
    python repro_scripts/compare_tb_logs.py --save comparison.png
"""

import argparse
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_BASE = "logs/rsl_rl/b2w_navigation_mdpo/2026-05-22_11-55-51/events.out.tfevents.1779465360.guacamole.1548052.0"
DEFAULT_ABLATION = "logs/rsl_rl/b2w_navigation_mdpo/2026-06-08_18-26-06/events.out.tfevents.1780957568.guacamole.1568647.0"


PLOT_GROUPS = [
    {"title": "Mean Episode Length", "filter": "Train/mean_episode_length"},
    {"title": "Mean Reward", "filter": "Train/mean_reward"},
    {"title": "Robot Goal", "filter": "Metrics/robot_goal/"},
    {"title": "Value Function", "filter": "Loss/value_function"},
    {"title": "KL Divergence", "filter": "Loss/kl_divergence"},
    {"title": "Reach Goal XY Tight", "filter": "Episode_Reward/reach_goal_xy_tight"},
    {"title": "Episode Termination", "filter": "Episode_Termination/time_out"},
]

COLS = 3


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


def plot_comparison(base_ea, base_tags, ablation_ea, ablation_tags, base_name, ablation_name, save_path=None):
    import matplotlib.pyplot as plt

    rows = (len(PLOT_GROUPS) + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(7 * COLS, 3 * rows), squeeze=False)

    for idx, group in enumerate(PLOT_GROUPS):
        ax = axes[idx // COLS][idx % COLS]
        filt = group["filter"]

        base_matched = find_matching_tags(base_tags, filt)
        ablation_matched = find_matching_tags(ablation_tags, filt)
        all_subtags = sorted(set(base_matched) | set(ablation_matched))

        if not all_subtags:
            ax.set_title(f"{group['title']} (no data)", fontsize=11)
            ax.set_visible(False)
            continue

        for tag in all_subtags:
            # Strip common prefix for legend
            short = tag.split("/")[-1] if "/" in tag else tag

            if tag in base_matched:
                events = base_ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                label = f"{base_name}: {short}"
                ax.plot(steps, values, label=label, linewidth=1.2)

            if tag in ablation_matched:
                events = ablation_ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                label = f"{ablation_name}: {short}"
                ax.plot(steps, values, label=label, linewidth=1.2, linestyle="--")

        ax.set_title(group["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(PLOT_GROUPS), rows * COLS):
        axes[idx // COLS][idx % COLS].set_visible(False)

    fig.suptitle(f"{base_name} vs {ablation_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare tensorboard logs from two runs.")
    parser.add_argument("--base", type=str, default=DEFAULT_BASE, help="Path to base run events file")
    parser.add_argument("--ablation", type=str, default=DEFAULT_ABLATION, help="Path to ablation run events file")
    parser.add_argument("--base-name", type=str, default="base", help="Display name for base run")
    parser.add_argument("--ablation-name", type=str, default="ablation", help="Display name for ablation run")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    print(f"Loading base run: {args.base}")
    base_ea, base_tags = load_events(args.base, args.base_name)
    print(f"  {len(base_tags)} tags found")

    print(f"Loading ablation run: {args.ablation}")
    ablation_ea, ablation_tags = load_events(args.ablation, args.ablation_name)
    print(f"  {len(ablation_tags)} tags found")

    plot_comparison(base_ea, base_tags, ablation_ea, ablation_tags, args.base_name, args.ablation_name, save_path=args.save)


if __name__ == "__main__":
    main()

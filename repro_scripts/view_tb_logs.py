#!/usr/bin/env python3
"""View and plot tensorboard event logs.

Usage:
    # Summary table of all metrics
    python scripts/view_tb_logs.py <log_dir>

    # Plot all metrics
    python scripts/view_tb_logs.py <log_dir> --plot

    # Plot specific tags (substring match)
    python scripts/view_tb_logs.py <log_dir> --plot --tag success_rate
    python scripts/view_tb_logs.py <log_dir> --plot --tag reward --tag success

    # Print full history for a tag
    python scripts/view_tb_logs.py <log_dir> --tag success_rate

    # Save plot to file instead of showing
    python scripts/view_tb_logs.py <log_dir> --plot --save plots.png
"""

import argparse
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_events(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = sorted(ea.Tags()["scalars"])
    if not tags:
        print(f"No scalar data found in {log_dir}")
        sys.exit(1)
    return ea, tags


def filter_tags(tags, filters):
    if not filters:
        return tags
    matched = []
    for t in tags:
        if any(f.lower() in t.lower() for f in filters):
            matched.append(t)
    return matched


def print_summary(ea, tags, last=None):
    print(f"{'Tag':<55} {'Points':>6}  {'First':>10}  {'Last':>10}  {'Step':>6}")
    print(f"{'-'*55} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*6}")
    for tag in tags:
        events = ea.Scalars(tag)
        if last:
            events = events[-last:]
        print(f"{tag:<55} {len(events):>6}  {events[0].value:>10.4f}  {events[-1].value:>10.4f}  {events[-1].step:>6}")


def print_history(ea, tags):
    for tag in tags:
        events = ea.Scalars(tag)
        print(f"\n{tag} ({len(events)} points):")
        print(f"{'Step':>8}  {'Value':>12}")
        print(f"{'----':>8}  {'-----':>12}")
        for e in events:
            print(f"{e.step:>8}  {e.value:>12.4f}")


def plot_tags(ea, tags, save_path=None):
    import matplotlib.pyplot as plt

    # Group tags by prefix (e.g. Episode_Reward, Loss, Metrics, etc.)
    groups = {}
    for tag in tags:
        prefix = tag.split("/")[0] if "/" in tag else "Other"
        groups.setdefault(prefix, []).append(tag)

    n = len(groups)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)

    for idx, (prefix, group_tags) in enumerate(sorted(groups.items())):
        ax = axes[idx // cols][idx % cols]
        for tag in group_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            label = tag.split("/", 1)[1] if "/" in tag else tag
            ax.plot(steps, values, label=label, linewidth=1)
        ax.set_title(prefix, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"Training Metrics ({tags[0].split('/')[0]}...)", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="View and plot tensorboard event logs.")
    parser.add_argument("log_dir", help="Path to the run directory containing events.out.tfevents.*")
    parser.add_argument("--tag", type=str, action="append", default=None,
                        help="Filter tags by substring (can be repeated). Without --plot, prints full history.")
    parser.add_argument("--last", type=int, default=None, help="Show only the last N points per tag")
    parser.add_argument("--plot", action="store_true", help="Plot metrics with matplotlib")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    ea, all_tags = load_events(args.log_dir)
    matched_tags = filter_tags(all_tags, args.tag)

    if not matched_tags:
        print(f"No tags matching {args.tag}. Available tags:")
        for t in all_tags:
            print(f"  {t}")
        sys.exit(1)

    if args.plot or args.save:
        plot_tags(ea, matched_tags, save_path=args.save)
    elif args.tag:
        print_history(ea, matched_tags)
    else:
        print(f"Run: {args.log_dir}\n")
        print_summary(ea, matched_tags, last=args.last)


if __name__ == "__main__":
    main()

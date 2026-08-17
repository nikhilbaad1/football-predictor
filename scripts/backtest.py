#!/usr/bin/env python3
"""Walk-forward backtest against the bookmaker baseline.

    python scripts/backtest.py                    # train pooled, score E0
    python scripts/backtest.py --score-all        # score every league
    python scripts/backtest.py --refit-every 20   # slower, slightly sharper

Reading the output: log_loss is the headline (lower is better). Expect roughly
1.10 for a uniform guess, ~1.03 for base rates, and 0.96-0.99 for de-vigged
closing odds. Landing between base rates and the bookmaker is the realistic
target — beating the closing line is not, and a result that appears to do so is
far more likely a bug than an edge.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpp.config import DISPLAY_DIVISION, DIVISIONS  # noqa: E402
from fpp.db import load_matches  # noqa: E402
from fpp.evaluation import calibration_table, walk_forward  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-train", type=int, default=2000)
    parser.add_argument("--refit-every", type=int, default=40)
    parser.add_argument("--score-all", action="store_true",
                        help="score all leagues, not just the display one")
    parser.add_argument("--save", type=Path, help="write per-match predictions to CSV")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    matches = load_matches(divisions=list(DIVISIONS), played_only=True)
    if matches.empty:
        print("No matches found. Run scripts/ingest.py first.")
        return 1

    matches = matches.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    test_filter = None if args.score_all else (matches["division"] == DISPLAY_DIVISION)

    scope = "all leagues" if args.score_all else DIVISIONS[DISPLAY_DIVISION]
    print(f"training on {len(matches):,} matches · scoring {scope}")
    print(f"initial train {args.min_train:,} · refit every {args.refit_every}\n")

    result = walk_forward(
        matches,
        min_train=args.min_train,
        refit_every=args.refit_every,
        test_filter=test_filter,
    )

    print(result)

    m = result.metrics.set_index("model")
    if "baseline_bookmaker" in m.index and "blend (odds subset)" in m.index:
        gap = m.loc["blend (odds subset)", "log_loss"] - m.loc["baseline_bookmaker", "log_loss"]
        print(f"\ngap to bookmaker: {gap:+.4f} log-loss")
        if gap < 0:
            print("  Beating the closing line. Check for leakage before believing it.")
        else:
            print("  Behind the closing line, as expected.")

    print("\ncalibration (blend, pooled across outcomes):")
    print(
        calibration_table(
            result.predictions[["blend_H", "blend_D", "blend_A"]].to_numpy(),
            result.predictions["result"].to_numpy(),
        ).to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    if args.save:
        result.predictions.to_csv(args.save, index=False)
        print(f"\nwrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

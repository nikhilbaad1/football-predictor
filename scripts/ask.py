#!/usr/bin/env python3
"""Ask the analyst agent a question.

    python scripts/ask.py "How many times have Arsenal beaten Chelsea?"
    python scripts/ask.py "Why did Arsenal leave Highbury?" --show-tools

The agent chooses between the match database and the article corpus. Exact
facts - results, counts, probabilities, availability - are counted, not
retrieved. Descriptive text is retrieved, not invented. That split is ADR 0003,
and choosing correctly is the whole job.

Costs roughly 5 cents a question. It reads and never writes.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpp.agents.analyst import ask  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("--show-tools", action="store_true",
                        help="print which tools were called, and the route taken")
    args = parser.parse_args()

    result = ask(args.question)

    if args.show_tools:
        print(f"route: {result.route}  ·  tools: {', '.join(result.tools_used) or 'none'}")
        print(f"cost:  ${result.cost_usd:.4f}\n")

    if not result.answer:
        print("The agent produced no answer. Try rephrasing the question.")
        return 1

    for para in result.answer.split("\n"):
        print("\n".join(textwrap.wrap(para, width=88)) if para.strip() else "")

    print("\nProbabilistic estimates, not betting advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

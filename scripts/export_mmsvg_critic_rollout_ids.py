#!/usr/bin/env python3
"""CLI wrapper for the deterministic MMSVG critic-rollout selector.

Run ``python scripts/export_mmsvg_critic_rollout_ids.py --help`` for the full
contract.  The command is intentionally fail-closed and refuses to replace an
existing output directory.
"""

from svg_agentic_slm.data.mmsvg_critic_selection import main

if __name__ == "__main__":
    raise SystemExit(main())

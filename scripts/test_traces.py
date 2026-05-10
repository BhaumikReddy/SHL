"""
scripts/test_traces.py

Evaluates the agent against the sample conversation traces in
GenAI_SampleConversations/ and reports Recall@10 per trace and overall.

Usage:
    # Start the server first:
    uvicorn app.main:app --reload --port 8888

    # Then in a separate terminal:
    python scripts/test_traces.py
"""

import glob
import os
import re
import sys
import time
from difflib import SequenceMatcher

import requests

BASE_URL = "http://127.0.0.1:8888/chat"
TRACES_DIR = os.path.join(os.path.dirname(__file__), "..", "GenAI_SampleConversations")

TURN_DELAY = 3   # seconds between turns within a trace
TRACE_DELAY = 5  # seconds between traces


def parse_trace(filepath: str) -> tuple[list[str], list[str]]:
    """
    Parse a conversation trace MD file.

    Returns:
        user_turns: list of user messages in order
        expected_names: list of assessment names from the LAST table before
                        end_of_conversation: true
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Extract user turns — lines starting with "> "
    user_turns = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("> "):
            user_turns.append(stripped[2:].strip())

    # Find all markdown tables in the document
    table_pattern = re.compile(
        r"(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)+)",
        re.MULTILINE,
    )
    tables = list(table_pattern.finditer(content))

    # Find the position of "end_of_conversation: **true**"
    eoc_match = re.search(r"end_of_conversation.*\*\*true\*\*", content)

    # Take the last table that appears BEFORE end_of_conversation: true
    expected_names = []
    if tables and eoc_match:
        candidate_tables = [t for t in tables if t.start() < eoc_match.start()]
        if candidate_tables:
            last_table = candidate_tables[-1].group()
            rows = last_table.strip().splitlines()

            # Parse header to find Name column index
            header = [cell.strip() for cell in rows[0].split("|") if cell.strip()]
            name_idx = next((i for i, h in enumerate(header) if h.lower() == "name"), None)

            if name_idx is not None:
                for row in rows[2:]:  # skip header + separator
                    cells = [cell.strip() for cell in row.split("|") if cell.strip()]
                    if len(cells) > name_idx:
                        name = cells[name_idx].strip()
                        if name and name != "#":
                            expected_names.append(name)

    return user_turns, expected_names


def replay_trace(user_turns: list[str]) -> list[dict]:
    """
    Replay ALL user turns against the live API.
    Only stops early if end_of_conversation is True.
    Adds delays between turns to avoid rate limits.
    Returns the final recommendations list from the last turn.
    """
    messages = []
    recommendations = []

    for i, turn in enumerate(user_turns):
        messages.append({"role": "user", "content": turn})

        try:
            resp = requests.post(BASE_URL, json={"messages": messages}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    [ERROR] Request failed: {e}")
            return recommendations

        reply = data.get("reply", "")
        current_recs = data.get("recommendations", []) or []
        end = data.get("end_of_conversation", False)

        # Always update recommendations with latest non-empty list
        if current_recs:
            recommendations = current_recs

        messages.append({"role": "assistant", "content": reply})

        print(f"    Turn {i + 1} done — recs so far: {len(recommendations)}, end: {end}")

        # Only stop early if agent says conversation is done
        if end:
            break

        # Wait between turns to avoid rate limit
        time.sleep(TURN_DELAY)

    return recommendations


def is_match(a: str, b: str) -> bool:
    """Fuzzy name match with > 85% similarity."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() > 0.85


def recall_at_k(returned: list[str], expected: list[str]) -> float:
    """Recall@10: fraction of expected names found in returned names using fuzzy matching."""
    if not expected:
        return 1.0
    matched = sum(1 for exp in expected if any(is_match(exp, got) for got in returned))
    return matched / len(expected)


def main():
    md_files = sorted(glob.glob(os.path.join(TRACES_DIR, "*.md")))

    if not md_files:
        print(f"No trace files found in {TRACES_DIR}")
        sys.exit(1)

    print(f"Found {len(md_files)} trace file(s)")
    print(f"Delays: {TURN_DELAY}s between turns, {TRACE_DELAY}s between traces\n")
    print(f"{'File':<14} {'Expected':>8} {'Matched':>8} {'Recall@10':>10}")
    print("-" * 46)

    recall_scores = []

    for idx, filepath in enumerate(md_files):
        filename = os.path.basename(filepath)
        print(f"\nRunning trace: {filename}")

        try:
            user_turns, expected_names = parse_trace(filepath)
        except Exception as e:
            print(f"  [PARSE ERROR] {e}")
            continue

        if not user_turns:
            print(f"  [SKIP] No user turns found")
            continue

        try:
            recommendations = replay_trace(user_turns)
        except Exception as e:
            print(f"  [REPLAY ERROR] {e}")
            continue

        returned_names = [r["name"] for r in recommendations]

        matched = sum(
            1 for exp in expected_names
            if any(is_match(exp, got) for got in returned_names)
        )
        score = recall_at_k(returned_names, expected_names)
        recall_scores.append(score)

        print(f"\n{filename:<14} {len(expected_names):>8} {matched:>8} {score:>10.2f}")
        print(f"  Expected : {expected_names}")
        print(f"  Returned : {returned_names}")

        # Wait between traces (skip delay after last trace)
        if idx < len(md_files) - 1:
            print(f"\n  Waiting {TRACE_DELAY}s before next trace...")
            time.sleep(TRACE_DELAY)

    print("\n" + "-" * 46)
    if recall_scores:
        mean_recall = sum(recall_scores) / len(recall_scores)
        print(f"{'Mean Recall@10':<30} {mean_recall:>10.2f}")
    else:
        print("No traces scored.")


if __name__ == "__main__":
    main()
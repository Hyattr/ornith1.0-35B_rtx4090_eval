"""Phase 2: reasoning トークンの分析。

A) max_tokens スイープ — 思考が枠を食い潰して最終回答が空になる境界を測る。
B) 難易度別の思考量 — 課題の難しさに対して思考トークンがどう増えるか。

reasoning モデル特有の挙動で、記事の中心に据える部分。
"""

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmclient as lm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BUDGETS = [64, 128, 256, 512, 1024, 2048]
N_SWEEP = 5
N_LADDER = 3

# A) 予算スイープ用。難易度が全く違う2問を同じ予算で比べる
SWEEP_TASKS = {
    "trivial": "Reply with exactly: PONG",
    "moderate": "A train travels 120 km in 1.5 hours, then 80 km in 0.5 hours. "
                "What is its average speed for the whole trip? Give just the number in km/h.",
}

# B) 難易度ラダー。1問あたりの思考量を難易度別に集計する
LADDER = {
    "1_trivial": [
        "Reply with exactly: PONG",
        "What is 2 + 2? Answer with just the number.",
        "Output the single word: apple",
    ],
    "2_easy": [
        "Write a Python one-liner that reverses a string.",
        "What is the capital of France? One word.",
        "Convert 45 degrees Celsius to Fahrenheit. Just the number.",
    ],
    "3_medium": [
        "Write a Python function that merges two sorted lists into one sorted list. "
        "Keep it under 15 lines.",
        "Explain the difference between a process and a thread in three sentences.",
        "Given a list of integers, write Python to find the second largest distinct value.",
    ],
    "4_hard": [
        "Implement an LRU cache in Python with O(1) get and put. Include brief comments.",
        "You have 8 balls, one is heavier. Using a balance scale twice, how do you find it? "
        "Explain the procedure.",
        "Write a Python function to detect a cycle in a directed graph and return the cycle "
        "if one exists.",
    ],
}


def sweep():
    """max_tokens を変えて空応答の発生率を測る。"""
    print("=== A) max_tokens スイープ ===")
    rows = []
    for task_name, prompt in SWEEP_TASKS.items():
        for budget in BUDGETS:
            empties = 0
            for i in range(N_SWEEP):
                r = lm.chat([{"role": "user", "content": prompt}],
                            max_tokens=budget, tag=f"p2a_{task_name}_{budget}")
                if r["empty_answer"]:
                    empties += 1
                rows.append({
                    "task": task_name, "max_tokens": budget, "run": i + 1,
                    "reasoning_tokens": r["reasoning_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "answer_chars": len(r["content"].strip()),
                    "finish_reason": r["finish_reason"],
                    "empty_answer": int(r["empty_answer"]),
                })
            rate = empties / N_SWEEP * 100
            print(f"  {task_name:>9} max_tokens={budget:>5} 空応答率 {rate:5.0f}% ({empties}/{N_SWEEP})")
    return rows


def ladder():
    """難易度別に思考トークン量を集計する。"""
    print("\n=== B) 難易度別の思考量 ===")
    rows = []
    for level, prompts in LADDER.items():
        for qi, prompt in enumerate(prompts):
            for i in range(N_LADDER):
                r = lm.chat([{"role": "user", "content": prompt}],
                            max_tokens=2048, tag=f"p2b_{level}_{qi}")
                rows.append({
                    "level": level, "question": qi, "run": i + 1,
                    "reasoning_tokens": r["reasoning_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "answer_tokens": (r["completion_tokens"] or 0) - (r["reasoning_tokens"] or 0),
                    "answer_chars": len(r["content"].strip()),
                    "finish_reason": r["finish_reason"],
                    "empty_answer": int(r["empty_answer"]),
                })
        vals = [x["reasoning_tokens"] for x in rows
                if x["level"] == level and x["reasoning_tokens"] is not None]
        print(f"  {level:>10} 思考トークン 中央値 {statistics.median(vals):7.0f} "
              f"(最小 {min(vals)} / 最大 {max(vals)}, n={len(vals)})")
    return rows


def save(name, rows, fields):
    out = RESULTS / name
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"保存: {out}")


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    snap = lm.config_snapshot()
    print("構成:", json.dumps(snap, ensure_ascii=False))
    lm.warmup(2)

    a = sweep()
    save("phase2a_budget_sweep.csv", a,
         ["task", "max_tokens", "run", "reasoning_tokens", "completion_tokens",
          "answer_chars", "finish_reason", "empty_answer"])

    b = ladder()
    save("phase2b_difficulty.csv", b,
         ["level", "question", "run", "reasoning_tokens", "completion_tokens",
          "answer_tokens", "answer_chars", "finish_reason", "empty_answer"])

    summary = {"config": snap, "budget_sweep": {}, "difficulty": {}}
    for task in SWEEP_TASKS:
        summary["budget_sweep"][task] = {
            str(bud): {
                "empty_rate_pct": round(
                    sum(r["empty_answer"] for r in a
                        if r["task"] == task and r["max_tokens"] == bud) / N_SWEEP * 100),
                "median_reasoning": statistics.median(
                    [r["reasoning_tokens"] for r in a
                     if r["task"] == task and r["max_tokens"] == bud
                     and r["reasoning_tokens"] is not None]),
            } for bud in BUDGETS
        }
    for level in LADDER:
        vals = [r["reasoning_tokens"] for r in b
                if r["level"] == level and r["reasoning_tokens"] is not None]
        summary["difficulty"][level] = {
            "median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals),
        }
    (RESULTS / "phase2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {RESULTS / 'phase2_summary.json'}")


if __name__ == "__main__":
    run()

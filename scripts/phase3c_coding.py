"""Phase 3-③: コード生成タスクを2審査員で採点する。

ornith で回答を生成し、Claude と Gemini が同一ルーブリックで独立に採点する。
一致率そのものが「採点の信頼性」を示す指標になる。

max_tokens は 4096。Phase 2 で難問の 6/9 が 2048 の上限に張り付き、
うち3回が空応答になったため、2048 では約3分の1を取りこぼす。
"""

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judges
import lmclient as lm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"

MAX_TOKENS = 4096

TASKS = [
    {"id": "t01_duration", "level": "easy", "prompt":
     "Write a Python function `parse_duration(s: str) -> int` that converts a "
     "duration string like '1h30m', '45s', or '2h' into a number of seconds. "
     "Support the units h, m and s in any combination and order. Raise "
     "ValueError with a clear message for input that is empty, has an unknown "
     "unit, or has no digits."},
    {"id": "t02_merge", "level": "easy", "prompt":
     "Write a Python function `merge_sorted(a: list[int], b: list[int]) -> list[int]` "
     "that merges two already-sorted lists into one sorted list without using "
     "sorted() or list.sort(). It must run in O(len(a)+len(b)) and handle empty "
     "inputs."},
    {"id": "t03_lru", "level": "medium", "prompt":
     "Implement an LRU cache in Python as a class `LRUCache` with `__init__(self, "
     "capacity)`, `get(key)` and `put(key, value)`. Both operations must be O(1). "
     "`get` returns None for a missing key. Evict the least recently used entry "
     "when capacity is exceeded; `get` counts as a use."},
    {"id": "t04_findbug", "level": "medium", "prompt":
     "The function below is supposed to remove every item with a negative value "
     "from the dict, but it behaves incorrectly. Identify the bug, explain why it "
     "happens, and give a corrected version.\n\n"
     "def drop_negatives(d):\n"
     "    for key in d:\n"
     "        if d[key] < 0:\n"
     "            del d[key]\n"
     "    return d"},
    {"id": "t05_retry", "level": "medium", "prompt":
     "Write a Python decorator `retry` that re-runs the wrapped function on "
     "exception. It must accept `max_attempts`, `base_delay` and `exceptions` "
     "(a tuple of exception types), use exponential backoff with jitter, re-raise "
     "the last exception when attempts are exhausted, and preserve the wrapped "
     "function's name and docstring."},
    {"id": "t06_flatten", "level": "medium", "prompt":
     "Write `flatten(d: dict, sep: str = '.') -> dict` that flattens a nested dict "
     "into dotted keys, e.g. {'a': {'b': 1}} becomes {'a.b': 1}. Lists must be "
     "indexed, so {'a': [10, 20]} becomes {'a.0': 10, 'a.1': 20}. Empty dicts and "
     "empty lists must be preserved as their own leaf values rather than "
     "disappearing."},
    {"id": "t07_bucket", "level": "medium", "prompt":
     "Implement a token-bucket rate limiter class `RateLimiter(rate, capacity)` "
     "in Python, where `rate` is tokens added per second. Provide "
     "`allow(cost: int = 1) -> bool` that consumes tokens if available. Refill "
     "lazily based on elapsed time rather than using a background thread, and "
     "never let the bucket exceed capacity."},
    {"id": "t08_cycle", "level": "hard", "prompt":
     "Write a Python function `find_cycle(graph: dict[str, list[str]]) -> "
     "list[str] | None` that detects a cycle in a directed graph given as an "
     "adjacency map. Return the nodes of one cycle in order if a cycle exists, "
     "otherwise None. It must handle self-loops, disconnected components, and "
     "nodes that appear only as targets."},
    {"id": "t09_race", "level": "hard", "prompt":
     "The counter below loses increments when used from multiple threads. Explain "
     "precisely which interleaving causes the loss, then give a corrected "
     "implementation. Explain why making only `value()` thread-safe would not be "
     "enough.\n\n"
     "class Counter:\n"
     "    def __init__(self):\n"
     "        self.n = 0\n"
     "    def increment(self):\n"
     "        self.n += 1\n"
     "    def value(self):\n"
     "        return self.n"},
    {"id": "t10_diff", "level": "hard", "prompt":
     "Write a Python function `diff(a: list[str], b: list[str]) -> list[tuple[str, "
     "str]]` that produces a line diff using the longest common subsequence. Each "
     "returned tuple is (op, line) where op is ' ' for unchanged, '-' for removed "
     "and '+' for added. The output, read in order, must reconstruct both inputs."},
    {"id": "t11_median", "level": "hard", "prompt":
     "Implement a streaming median class `MedianStream` in Python with `add(x)` "
     "and `median()`. Use two heaps so that `add` is O(log n) and `median` is "
     "O(1). Return the average of the two middle values for an even count, and "
     "raise a clear error if `median()` is called before any value is added."},
    {"id": "t12_bsearch", "level": "hard", "prompt":
     "The binary search below is meant to return the index of the FIRST "
     "occurrence of target in a sorted list that may contain duplicates, or -1 if "
     "absent. It does not always do so. Identify the defect, give a concrete "
     "input that triggers it, and provide a corrected implementation.\n\n"
     "def first_index(xs, target):\n"
     "    lo, hi = 0, len(xs) - 1\n"
     "    while lo <= hi:\n"
     "        mid = (lo + hi) // 2\n"
     "        if xs[mid] == target:\n"
     "            return mid\n"
     "        elif xs[mid] < target:\n"
     "            lo = mid + 1\n"
     "        else:\n"
     "            hi = mid - 1\n"
     "    return -1"},
]


def generate():
    """ornith に全タスクを解かせる。"""
    print("=== 回答の生成 (ornith) ===")
    answers = []
    for t in TASKS:
        r = lm.chat([{"role": "user", "content": t["prompt"]}],
                    max_tokens=MAX_TOKENS, tag=f"p3c_{t['id']}")
        answers.append({**t, "answer": r["content"],
                        "finish_reason": r["finish_reason"],
                        "reasoning_tokens": r["reasoning_tokens"],
                        "completion_tokens": r["completion_tokens"],
                        "empty_answer": r["empty_answer"]})
        flag = " ⚠️空応答" if r["empty_answer"] else ""
        print(f"  {t['id']:<14} {t['level']:<7} 思考 {r['reasoning_tokens']:>5} / "
              f"回答 {len(r['content']):>5}字  {r['finish_reason']}{flag}")
    return answers


def score(answers):
    """2審査員で独立に採点する。"""
    print("\n=== 採点 (Claude / Gemini) ===")
    rows = []
    for a in answers:
        row = {"case": a["id"], "level": a["level"],
               "reasoning_tokens": a["reasoning_tokens"],
               "answer_chars": len(a["answer"]),
               "finish_reason": a["finish_reason"]}
        for name, fn in judges.JUDGES.items():
            try:
                s = fn(a["prompt"], a["answer"], tag=f"judge_{name}_{a['id']}")
                for k in ("correctness", "completeness", "code_quality", "overall"):
                    row[f"{name}_{k}"] = s[k]
                row[f"{name}_justification"] = s["justification"]
            except Exception as e:
                print(f"    {name} 採点失敗 ({a['id']}): {type(e).__name__} {str(e)[:120]}")
                row[f"{name}_overall"] = None
        c, g = row.get("claude_overall"), row.get("gemini_overall")
        agree = "―" if None in (c, g) else ("一致" if c == g else f"差{abs(c-g)}")
        print(f"  {a['id']:<14} Claude {c} / Gemini {g}   {agree}")
        rows.append(row)
    return rows


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PROMPTS.mkdir(parents=True, exist_ok=True)
    (PROMPTS / "phase3c_tasks.json").write_text(
        json.dumps(TASKS, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROMPTS / "judge_rubric.md").write_text(judges.RUBRIC, encoding="utf-8")

    snap = lm.config_snapshot()
    print("構成:", json.dumps(snap, ensure_ascii=False))
    lm.warmup(2)

    rows = score(generate())

    fields = ["case", "level", "reasoning_tokens", "answer_chars", "finish_reason"]
    for name in judges.JUDGES:
        fields += [f"{name}_{k}" for k in
                   ("correctness", "completeness", "code_quality", "overall")]
    fields += [f"{name}_justification" for name in judges.JUDGES]
    out = RESULTS / "phase3c_coding.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        f.write("# " + json.dumps(snap, ensure_ascii=False) + "\n")
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n保存: {out}")

    paired = [r for r in rows
              if r.get("claude_overall") is not None and r.get("gemini_overall") is not None]
    exact = sum(1 for r in paired if r["claude_overall"] == r["gemini_overall"])
    within1 = sum(1 for r in paired if abs(r["claude_overall"] - r["gemini_overall"]) <= 1)
    disagreements = sorted(
        (r for r in paired if r["claude_overall"] != r["gemini_overall"]),
        key=lambda r: -abs(r["claude_overall"] - r["gemini_overall"]))

    summary = {
        "config": snap, "max_tokens": MAX_TOKENS, "tasks": len(TASKS),
        "judges": {"claude": judges.CLAUDE_MODEL, "gemini": judges.GEMINI_MODEL},
        "mean_overall": {
            name: round(statistics.mean([r[f"{name}_overall"] for r in paired]), 2)
            for name in judges.JUDGES},
        "agreement": {
            "pairs": len(paired),
            "exact_pct": round(exact / len(paired) * 100, 1) if paired else None,
            "within_1_pct": round(within1 / len(paired) * 100, 1) if paired else None,
        },
        "disagreements": [
            {"case": r["case"], "level": r["level"],
             "claude": r["claude_overall"], "gemini": r["gemini_overall"]}
            for r in disagreements],
        "by_level": {
            lv: {name: round(statistics.mean(
                    [r[f"{name}_overall"] for r in paired if r["level"] == lv]), 2)
                 for name in judges.JUDGES}
            for lv in ("easy", "medium", "hard")
            if any(r["level"] == lv for r in paired)},
    }
    (RESULTS / "phase3c_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 集計 ===")
    for name in judges.JUDGES:
        print(f"  {name:>7} の平均スコア : {summary['mean_overall'][name]}")
    print(f"  完全一致            : {summary['agreement']['exact_pct']}%  "
          f"({exact}/{len(paired)})")
    print(f"  ±1 以内             : {summary['agreement']['within_1_pct']}%")
    if disagreements:
        print("  不一致だった項目    :")
        for d in summary["disagreements"]:
            print(f"    - {d['case']} ({d['level']}): Claude {d['claude']} / Gemini {d['gemini']}")


if __name__ == "__main__":
    run()

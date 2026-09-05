"""Phase 3-①: ツール呼び出し精度の計測。

ornith はエージェント型コーディング特化モデルなので、ここが本領。
判定は機械採点で、LLM審査員を使わない。バイアスの注釈が要らないぶん
記事の数値として最も強い。

3種類の失敗を区別して測る:
  - 呼ぶべき場面で呼ばない / 別のツールを呼ぶ (ツール選択)
  - 呼ぶべきでない場面で呼ぶ                  (誤検知)
  - 呼ぶが引数が違う                          (引数精度)
"""

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmclient as lm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

N = 3           # 各シナリオの試行回数
MAX_TOKENS = 2048

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        }, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_files",
        "description": "Search for files matching a glob pattern in the repository.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. *.py"},
            "directory": {"type": "string", "description": "Directory to search in"},
        }, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run the test suite for a given path and return the results.",
        "parameters": {"type": "object", "properties": {
            "test_path": {"type": "string", "description": "Path to the test file or directory"},
        }, "required": ["test_path"]}}},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email. This action is irreversible.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"},
        }, "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression and return the numeric result.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. (3+4)*2"},
        }, "required": ["expression"]}}},
]

# expect が None のケースは「ツールを呼ばないのが正解」。
# 引数は完全一致ではなく、要点が含まれるかで判定する（表記ゆれを許容）。
CASES = [
    # --- 呼ぶべき: ツール選択 + 引数 ---
    {"id": "call_weather", "kind": "should_call",
     "prompt": "What's the weather like in Kyoto right now?",
     "expect": {"tool": "get_weather", "args_contain": {"city": "kyoto"}}},
    {"id": "call_weather_unit", "kind": "should_call",
     "prompt": "Tell me the current temperature in Boston in fahrenheit.",
     "expect": {"tool": "get_weather", "args_contain": {"city": "boston", "unit": "fahrenheit"}}},
    {"id": "call_search_py", "kind": "should_call",
     "prompt": "Find all the Python files in the src directory.",
     "expect": {"tool": "search_files", "args_contain": {"pattern": ".py"}}},
    {"id": "call_search_yaml", "kind": "should_call",
     "prompt": "Locate every YAML config file in this repo.",
     "expect": {"tool": "search_files", "args_contain": {"pattern": "y"}}},
    {"id": "call_tests", "kind": "should_call",
     "prompt": "Run the tests in tests/test_parser.py and tell me if they pass.",
     "expect": {"tool": "run_tests", "args_contain": {"test_path": "tests/test_parser.py"}}},
    {"id": "call_tests_dir", "kind": "should_call",
     "prompt": "Please execute the whole test suite under the tests directory.",
     "expect": {"tool": "run_tests", "args_contain": {"test_path": "tests"}}},
    {"id": "call_calc", "kind": "should_call",
     "prompt": "Compute (1234 * 5678) + 91011 exactly.",
     "expect": {"tool": "calculate", "args_contain": {"expression": "1234"}}},
    {"id": "call_email", "kind": "should_call",
     "prompt": "Send an email to alice@example.com with the subject 'Build failed' "
               "and a body explaining that the nightly build failed.",
     "expect": {"tool": "send_email", "args_contain": {"to": "alice@example.com",
                                                       "subject": "build failed"}}},

    # --- 呼ぶべきでない: 誤検知の検出 ---
    {"id": "no_greeting", "kind": "should_not_call",
     "prompt": "Hi there! How are you doing today?", "expect": None},
    {"id": "no_general_knowledge", "kind": "should_not_call",
     "prompt": "What is the difference between a list and a tuple in Python?",
     "expect": None},
    {"id": "no_opinion", "kind": "should_not_call",
     "prompt": "In two sentences, why do people prefer static typing in large codebases?",
     "expect": None},
    {"id": "no_definition", "kind": "should_not_call",
     "prompt": "Explain what a race condition is. Do not use any tools.",
     "expect": None},
    {"id": "no_trivial_math", "kind": "should_not_call",
     "prompt": "What is 2 + 2? Just answer directly without calling anything.",
     "expect": None},
    {"id": "no_weather_hypothetical", "kind": "should_not_call",
     "prompt": "In general, what causes fog to form near coastlines?",
     "expect": None},

    # --- 紛らわしい: 似たツール間の選択 ---
    {"id": "pick_search_not_tests", "kind": "should_call",
     "prompt": "I want to know which test files exist. Just list them, don't execute anything.",
     "expect": {"tool": "search_files", "args_contain": {}}},
    {"id": "pick_tests_not_search", "kind": "should_call",
     "prompt": "Check whether tests/test_api.py currently passes.",
     "expect": {"tool": "run_tests", "args_contain": {"test_path": "tests/test_api.py"}}},
    {"id": "pick_calc_not_direct", "kind": "should_call",
     "prompt": "I need the exact value of 98765 * 43210. Use the calculator, precision matters.",
     "expect": {"tool": "calculate", "args_contain": {"expression": "98765"}}},

    # --- 破壊的操作: 明示されていない場合は呼ばないのが望ましい ---
    {"id": "no_email_vague", "kind": "should_not_call",
     "prompt": "I should probably let the team know about the outage at some point.",
     "expect": None},
    {"id": "no_email_draft", "kind": "should_not_call",
     "prompt": "Draft the text of an email about the outage, but do not send anything.",
     "expect": None},
    {"id": "call_weather_multi", "kind": "should_call",
     "prompt": "What is the weather in Osaka?",
     "expect": {"tool": "get_weather", "args_contain": {"city": "osaka"}}},
]


def judge(case, result):
    """1回の実行を採点する。返り値は (結果ラベル, 詳細)。"""
    calls = result["tool_calls"]
    expect = case["expect"]

    if expect is None:
        return ("correct" if not calls else "false_positive",
                calls[0]["function"]["name"] if calls else "")

    if not calls:
        return ("missed_call", "")

    name = calls[0]["function"]["name"]
    if name != expect["tool"]:
        return ("wrong_tool", name)

    # ツールは合っている。引数を確認する。
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except (json.JSONDecodeError, TypeError):
        return ("bad_arg_json", calls[0]["function"]["arguments"][:80])

    for key, needle in expect["args_contain"].items():
        actual = str(args.get(key, "")).lower()
        if needle.lower() not in actual:
            return ("wrong_args", f"{key}={args.get(key)!r} (期待: …{needle}…)")
    return ("correct", name)


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    snap = lm.config_snapshot()
    print("構成:", json.dumps(snap, ensure_ascii=False))
    lm.warmup(2)

    rows = []
    for case in CASES:
        outcomes = []
        for i in range(N):
            r = lm.chat([{"role": "user", "content": case["prompt"]}],
                        max_tokens=MAX_TOKENS, tools=TOOLS, tag=f"p3a_{case['id']}")
            verdict, detail = judge(case, r)
            outcomes.append(verdict)
            rows.append({
                "case": case["id"], "kind": case["kind"], "run": i + 1,
                "verdict": verdict, "detail": detail,
                "called": calls[0]["function"]["name"] if (calls := r["tool_calls"]) else "",
                "finish_reason": r["finish_reason"],
                "reasoning_tokens": r["reasoning_tokens"],
            })
        ok = sum(1 for o in outcomes if o == "correct")
        mark = "OK " if ok == N else ("--" if ok == 0 else "~~")
        print(f"  {mark} {case['id']:<24} {ok}/{N}  {Counter(outcomes).most_common()}")

    out = RESULTS / "phase3a_toolcalling.csv"
    fields = ["case", "kind", "run", "verdict", "detail", "called",
              "finish_reason", "reasoning_tokens"]
    with out.open("w", newline="", encoding="utf-8") as f:
        f.write("# " + json.dumps(snap, ensure_ascii=False) + "\n")
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n保存: {out}")

    total = len(rows)
    correct = sum(1 for r in rows if r["verdict"] == "correct")
    should_call = [r for r in rows if r["kind"] == "should_call"]
    should_not = [r for r in rows if r["kind"] == "should_not_call"]

    summary = {
        "config": snap,
        "n_per_case": N,
        "total_runs": total,
        "overall_accuracy_pct": round(correct / total * 100, 1),
        "should_call": {
            "runs": len(should_call),
            "accuracy_pct": round(
                sum(1 for r in should_call if r["verdict"] == "correct")
                / len(should_call) * 100, 1),
            "breakdown": dict(Counter(r["verdict"] for r in should_call)),
        },
        "should_not_call": {
            "runs": len(should_not),
            "accuracy_pct": round(
                sum(1 for r in should_not if r["verdict"] == "correct")
                / len(should_not) * 100, 1),
            "false_positive_pct": round(
                sum(1 for r in should_not if r["verdict"] == "false_positive")
                / len(should_not) * 100, 1),
        },
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "median_reasoning_tokens": statistics.median(
            [r["reasoning_tokens"] for r in rows if r["reasoning_tokens"] is not None]),
    }
    (RESULTS / "phase3a_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 集計 ===")
    print(f"  全体精度            : {summary['overall_accuracy_pct']}%  ({correct}/{total})")
    print(f"  呼ぶべき場面        : {summary['should_call']['accuracy_pct']}%")
    print(f"  呼ぶべきでない場面  : {summary['should_not_call']['accuracy_pct']}%"
          f"  (誤検知 {summary['should_not_call']['false_positive_pct']}%)")
    print(f"  内訳                : {summary['verdicts']}")
    print(f"  思考トークン中央値  : {summary['median_reasoning_tokens']:.0f}")


if __name__ == "__main__":
    run()

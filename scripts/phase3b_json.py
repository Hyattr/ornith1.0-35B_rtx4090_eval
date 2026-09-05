"""Phase 3-②: JSON スキーマ準拠率の計測。

同じ要求を繰り返し投げて、指定スキーマに沿った JSON を返せる割合を測る。
実務でローカルLLMを組み込む際に最も効く指標のひとつ。

2つのモードを比較する:
  prompt  … プロンプトで「このスキーマに従え」と指示するだけ
  schema  … response_format でスキーマを強制する (OpenAI互換の機能)

さらに2段階で判定する:
  strict  … 応答をそのまま json.loads() できるか
  lenient … ```json フェンスを剥がせば読めるか
フェンス付きで返す挙動は「モデルの失敗」ではなく「クライアント側で
剥がせば済む」問題なので、分けて数えないと実態を見誤る。
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmclient as lm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

N = 30
MAX_TOKENS = 2048

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "estimated_hours": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "assignee": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["name", "email"],
            "additionalProperties": False,
        },
        "blocked": {"type": "boolean"},
    },
    "required": ["title", "priority", "estimated_hours", "tags", "assignee", "blocked"],
    "additionalProperties": False,
}

BUG_REPORT = """\
The CSV importer crashes when a file has a UTF-8 BOM. Reported by Dana Wu
(dana.wu@example.com), who is picking it up. She thinks it is about half a
day of work. It is holding up the release, and it touches the parser and
the file-io layer. Nothing is blocking her from starting."""

PROMPT_ONLY = (
    "Convert the following bug report into a task object.\n\n"
    f"{BUG_REPORT}\n\n"
    "Respond with a single JSON object matching exactly this schema "
    "(no extra properties, no markdown fences, no commentary):\n"
    f"{json.dumps(SCHEMA, indent=2)}"
)

WITH_SCHEMA = (
    "Convert the following bug report into a task object.\n\n"
    f"{BUG_REPORT}"
)

FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def evaluate(text):
    """1件の応答を判定して dict を返す。"""
    out = {"strict_json": 0, "lenient_json": 0, "schema_valid": 0,
           "had_fence": 0, "error": ""}

    def check_schema(obj):
        try:
            jsonschema.validate(obj, SCHEMA)
            out["schema_valid"] = 1
        except jsonschema.ValidationError as e:
            out["error"] = e.message[:120]

    try:
        obj = json.loads(text)
        out["strict_json"] = out["lenient_json"] = 1
        check_schema(obj)
        return out
    except json.JSONDecodeError:
        pass

    m = FENCE.match(text)
    if m:
        out["had_fence"] = 1
        try:
            obj = json.loads(m.group(1))
            out["lenient_json"] = 1
            check_schema(obj)
            return out
        except json.JSONDecodeError as e:
            out["error"] = f"フェンス除去後もパース失敗: {e}"[:120]
            return out

    out["error"] = "JSONとして解釈できない"
    return out


def run_mode(mode):
    print(f"\n=== モード: {mode} ===")
    rows = []
    for i in range(N):
        kwargs = {}
        if mode == "prompt":
            content = PROMPT_ONLY
        else:
            content = WITH_SCHEMA
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "task", "strict": True, "schema": SCHEMA},
            }
        try:
            r = lm.chat([{"role": "user", "content": content}],
                        max_tokens=MAX_TOKENS, tag=f"p3b_{mode}", extra=kwargs)
        except lm.LMError as e:
            print(f"  [{i+1}/{N}] エラー: {str(e)[:120]}")
            rows.append({"mode": mode, "run": i + 1, "error": str(e)[:200]})
            continue

        ev = evaluate(r["content"])
        rows.append({"mode": mode, "run": i + 1, **ev,
                     "finish_reason": r["finish_reason"],
                     "reasoning_tokens": r["reasoning_tokens"],
                     "answer_chars": len(r["content"])})
        if (i + 1) % 10 == 0:
            ok = sum(x.get("schema_valid", 0) for x in rows)
            print(f"  [{i+1}/{N}] スキーマ適合 {ok}/{i+1}")
    return rows


def summarize(rows, mode):
    sub = [r for r in rows if r["mode"] == mode and "strict_json" in r]
    n = len(sub) or 1
    return {
        "runs": len(sub),
        "strict_json_pct": round(sum(r["strict_json"] for r in sub) / n * 100, 1),
        "lenient_json_pct": round(sum(r["lenient_json"] for r in sub) / n * 100, 1),
        "schema_valid_pct": round(sum(r["schema_valid"] for r in sub) / n * 100, 1),
        "markdown_fence_pct": round(sum(r["had_fence"] for r in sub) / n * 100, 1),
        "top_errors": [e for e, _ in Counter(
            r["error"] for r in sub if r["error"]).most_common(3)],
    }


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    snap = lm.config_snapshot()
    print("構成:", json.dumps(snap, ensure_ascii=False))
    lm.warmup(2)

    rows = run_mode("prompt") + run_mode("schema")

    out = RESULTS / "phase3b_json.csv"
    fields = ["mode", "run", "strict_json", "lenient_json", "schema_valid",
              "had_fence", "finish_reason", "reasoning_tokens", "answer_chars", "error"]
    with out.open("w", newline="", encoding="utf-8") as f:
        f.write("# " + json.dumps(snap, ensure_ascii=False) + "\n")
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n保存: {out}")

    summary = {"config": snap, "n_per_mode": N, "schema": SCHEMA,
               "prompt": summarize(rows, "prompt"),
               "schema_enforced": summarize(rows, "schema")}
    (RESULTS / "phase3b_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 集計 ===")
    print(f"{'':<22}{'そのままJSON':>12}{'フェンス除去後':>14}{'スキーマ適合':>13}{'フェンス付き':>13}")
    for label, key in [("プロンプト指示のみ", "prompt"), ("response_format強制", "schema_enforced")]:
        s = summary[key]
        print(f"{label:<20}{s['strict_json_pct']:>10}%{s['lenient_json_pct']:>13}%"
              f"{s['schema_valid_pct']:>12}%{s['markdown_fence_pct']:>12}%")
    for key in ("prompt", "schema_enforced"):
        if summary[key]["top_errors"]:
            print(f"\n{key} の主なエラー:")
            for e in summary[key]["top_errors"]:
                print(f"  - {e}")


if __name__ == "__main__":
    run()

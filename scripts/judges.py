"""2つの審査員 (Claude / Gemini) で同一ルーブリック採点を行う。

系統の異なる2社のモデルを使うのは冗長化のためではなく、**一致率という
検証可能な指標**を得るため。審査員が1つだと採点の妥当性を誰も検証できない。

公平性のための取り決め:
  - 両者に完全に同一のプロンプトを渡す
  - 両者とも構造化出力でスコアを受け取る（自由文パースの失敗を排除）
  - 採点対象がどのモデルの出力かは審査員に伝えない（盲検）
  - 生レスポンスは両者とも全件保存する

⚠️ 記事に書くべき非対称性: 評価対象の ornith は Gemma 4 の上に
post-training されている。Gemma は Google 系であり、Gemini との間には
Claude より近い血縁がありうる。自己強化バイアスが Gemini 側にだけ働く
可能性は否定できないため、スコアは平均せず両者を併記する。
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401  (import した時点で .env が読み込まれる)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CLAUDE_MODEL = "claude-opus-5"
GEMINI_MODEL = "gemini-3.8-flash"

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "completeness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "code_quality": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "overall": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "justification": {"type": "string"},
    },
    "required": ["correctness", "completeness", "code_quality", "overall", "justification"],
    "additionalProperties": False,
}

RUBRIC = """\
You are grading a single answer produced by an unnamed AI model. Grade only \
what is in front of you.

Score each criterion on a 1-5 integer scale:

- correctness   1 = wrong or does not run; 3 = works for the common case but has a \
real defect; 5 = correct, including the edge cases the task implies.
- completeness  1 = ignores most of what was asked; 3 = covers the main request but \
drops a stated requirement; 5 = every stated requirement is addressed.
- code_quality  1 = unreadable or misleading; 3 = works but is clumsy or poorly named; \
5 = clear, idiomatic, appropriately commented.
- overall       your single summary judgement, not a mechanical average.

Grading rules:
- Judge substance, not length. A short answer that fully solves the task scores \
higher than a long one that does not. Do not reward padding, restated requirements, \
or unrequested extra features.
- If the answer is empty or is only reasoning with no deliverable, score 1 across \
the board.
- justification: 2-3 sentences citing specific evidence from the answer. Name the \
concrete defect if you deduct points.

=== TASK GIVEN TO THE MODEL ===
{task}

=== MODEL'S ANSWER ===
{answer}
"""


def _save(tag, payload):
    DATA.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S_%f")
    (DATA / f"{tag}_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_prompt(task, answer):
    return RUBRIC.format(task=task, answer=answer if answer.strip() else "(empty response)")


def judge_claude(task, answer, tag="judge_claude"):
    """Claude で採点する。構造化出力でスコアを受け取る。"""
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
        messages=[{"role": "user", "content": build_prompt(task, answer)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Claude が採点を拒否: {resp.stop_details}")
    text = next(b.text for b in resp.content if b.type == "text")
    scores = json.loads(text)
    _save(tag, {"model": CLAUDE_MODEL, "scores": scores,
                "usage": {"input": resp.usage.input_tokens,
                          "output": resp.usage.output_tokens},
                "stop_reason": resp.stop_reason})
    return scores


def judge_gemini(task, answer, tag="judge_gemini"):
    """Gemini で採点する。構造化出力でスコアを受け取る。"""
    from google import genai

    client = genai.Client()
    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=build_prompt(task, answer),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": SCORE_SCHEMA,
        },
    )
    scores = json.loads(interaction.output_text)
    _save(tag, {"model": GEMINI_MODEL, "scores": scores})
    return scores


JUDGES = {"claude": judge_claude, "gemini": judge_gemini}


if __name__ == "__main__":
    # 疎通確認: 明らかに不完全な回答を渡して、両者が低いスコアを付けるか見る
    task = "Write a Python function `add(a, b)` that returns the sum of two integers."
    answer = "def add(a, b):\n    return a - b\n"
    for name, fn in JUDGES.items():
        try:
            s = fn(task, answer, tag=f"smoke_{name}")
            print(f"{name:>7}: overall={s['overall']} correctness={s['correctness']} "
                  f"| {s['justification'][:110]}")
        except Exception as e:
            print(f"{name:>7}: ERROR {type(e).__name__}: {str(e)[:200]}")

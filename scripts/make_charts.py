"""計測結果から記事用のグラフを生成する。

results/*.csv を読み、results/*.png を出力する。
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
ACCENT, ACCENT2 = "#2b6cb0", "#c05621"


def read_csv(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(lines))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def phase1_charts():
    rows = [r for r in read_csv("phase1_systems.csv") if not r.get("error")]
    if not rows:
        print("phase1: データなし")
        return
    by = defaultdict(list)
    for r in rows:
        by[int(r["target_tokens"])].append(r)
    xs = sorted(by)

    def med(target, key):
        vals = [fnum(r[key]) for r in by[target] if fnum(r[key]) is not None]
        return statistics.median(vals) if vals else None

    prompt = [med(x, "prompt_tokens") for x in xs]
    ttft = [med(x, "ttft_s") for x in xs]
    decode = [med(x, "decode_tok_s") for x in xs]
    prefill = [med(x, "prefill_tok_s") for x in xs]

    # 3指標はスケールが全く違うので、軸を共有せず個別のパネルに分ける。
    # decode は変動幅が1割弱しかないため、0起点にすると傾向が潰れる。
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.2))

    a1.plot(prompt, ttft, "o-", color=ACCENT, lw=2)
    a1.set_xlabel("入力トークン数 (実測)")
    a1.set_ylabel("TTFT (秒)")
    a1.set_title("① 入力長 vs 最初の1トークンまで")

    a2.plot(prompt, prefill, "s-", color=ACCENT2, lw=2)
    a2.set_xlabel("入力トークン数 (実測)")
    a2.set_ylabel("prefill 速度 (tok/s)")
    a2.set_title("② 入力処理スループット")
    a2.set_ylim(bottom=0)

    a3.plot(prompt, decode, "o-", color=ACCENT, lw=2)
    a3.set_xlabel("入力トークン数 (実測)")
    a3.set_ylabel("decode 速度 (tok/s)")
    a3.set_title("③ 生成速度")
    if decode[0] and decode[-1]:
        drop = (decode[0] - decode[-1]) / decode[0] * 100
        a3.annotate(f"{prompt[0]:,.0f}→{prompt[-1]:,.0f}トークンで {drop:.1f}% 低下",
                    xy=(0.5, 0.08), xycoords="axes fraction", ha="center", fontsize=9,
                    bbox={"boxstyle": "round,pad=0.4", "fc": "#f7fafc", "ec": "#cbd5e0"})

    fig.suptitle("Phase 1: システム指標 (Ornith-1.0-35B / UD-IQ4_XS / RTX 4090 / context 32768)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "phase1_speed.png", bbox_inches="tight")
    print("生成: phase1_speed.png")

    cache = read_csv("phase1_prompt_cache.csv")
    if cache:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        px = [fnum(r["prompt_tokens"]) for r in cache]
        idx = range(len(px))
        width = 0.38
        ax.bar([i - width / 2 for i in idx], [fnum(r["cold_ttft_s"]) for r in cache],
               width, label="cold (キャッシュなし)", color=ACCENT)
        ax.bar([i + width / 2 for i in idx], [fnum(r["warm_ttft_s"]) for r in cache],
               width, label="warm (キャッシュヒット)", color=ACCENT2)
        for i, r in enumerate(cache):
            ax.text(i, fnum(r["cold_ttft_s"]), f" {fnum(r['speedup']):.0f}倍", ha="center",
                    va="bottom", fontsize=9)
        ax.set_xticks(list(idx))
        ax.set_xticklabels([f"{int(v):,}" for v in px])
        ax.set_xlabel("入力トークン数")
        ax.set_ylabel("TTFT (秒)")
        ax.set_title("プロンプトキャッシュの効果\n同じ入力を繰り返すと TTFT が実態とかけ離れる")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "phase1_prompt_cache.png", bbox_inches="tight")
        print("生成: phase1_prompt_cache.png")


def phase2_charts():
    a = read_csv("phase2a_budget_sweep.csv")
    if a:
        # 易しい順に並べる（sorted だと moderate が先に来て直感に反する）
        order = ["trivial", "moderate"]
        present = {r["task"] for r in a}
        tasks = [t for t in order if t in present] + sorted(present - set(order))
        budgets = sorted({int(r["max_tokens"]) for r in a})
        fig, ax = plt.subplots(figsize=(7, 4.2))
        width = 0.38
        for i, task in enumerate(tasks):
            rates = []
            for b in budgets:
                sub = [r for r in a if r["task"] == task and int(r["max_tokens"]) == b]
                rates.append(sum(int(r["empty_answer"]) for r in sub) / len(sub) * 100 if sub else 0)
            xs = [j + (i - 0.5) * width for j in range(len(budgets))]
            ax.bar(xs, rates, width, label=task, color=[ACCENT, ACCENT2][i % 2])
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels(budgets)
        ax.set_xlabel("max_tokens")
        ax.set_ylabel("最終回答が空になった割合 (%)")
        ax.set_title("思考が予算を食い潰して回答が消える境界")
        ax.set_ylim(0, 105)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "phase2_empty_rate.png", bbox_inches="tight")
        print("生成: phase2_empty_rate.png")

    b = read_csv("phase2b_difficulty.csv")
    if b:
        levels = sorted({r["level"] for r in b})
        data = [[fnum(r["reasoning_tokens"]) for r in b
                 if r["level"] == lv and fnum(r["reasoning_tokens"]) is not None]
                for lv in levels]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bp = ax.boxplot(data, tick_labels=[l.split("_", 1)[1] for l in levels],
                        patch_artist=True, medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor(ACCENT)
            patch.set_alpha(0.6)
        ax.set_xlabel("課題の難易度")
        ax.set_ylabel("思考トークン数 (reasoning_tokens)")
        ax.set_title("難易度別の思考量 (max_tokens=2048)")
        # 上限に張り付いた実行は思考が打ち切られている。実際の必要量はこれより多い。
        ax.axhline(2048, color="crimson", ls="--", lw=1.4)
        ax.text(0.98, 2048, " max_tokens 上限", color="crimson", fontsize=8,
                va="bottom", ha="right", transform=ax.get_yaxis_transform())
        fig.tight_layout()
        fig.savefig(RESULTS / "phase2_difficulty.png", bbox_inches="tight")
        print("生成: phase2_difficulty.png")


def phase3_charts():
    rows = read_csv("phase3c_coding.csv")
    if not rows:
        print("phase3: データなし")
        return

    LEVEL_COLOR = {"easy": "#2f855a", "medium": ACCENT, "hard": ACCENT2}
    labels = [r["case"].split("_", 1)[1] for r in rows]
    reasoning = [fnum(r["reasoning_tokens"]) for r in rows]
    colors = [LEVEL_COLOR.get(r["level"], ACCENT) for r in rows]

    # ① 思考トークンと予算の関係。4096 で 5/12 が溢れたことを図で示す。
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(range(len(rows)), reasoning, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("思考トークン数")
    ax.set_title("課題ごとの思考量と max_tokens 予算")
    for budget, color, note in [(2048, "#a0aec0", "2048"), (4096, "crimson", "4096 (5/12が超過)")]:
        ax.axhline(budget, color=color, ls="--", lw=1.4)
        ax.text(0.995, budget, f" {note}", color=color, fontsize=8, va="bottom",
                ha="right", transform=ax.get_yaxis_transform())
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LEVEL_COLOR.values()]
    ax.legend(handles, LEVEL_COLOR.keys(), title="出題時の難易度ラベル", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "phase3_reasoning_budget.png", bbox_inches="tight")
    print("生成: phase3_reasoning_budget.png")

    # ② 2審査員のスコア比較。平均せず並べることで一致/不一致を見せる。
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    width = 0.38
    xs = range(len(rows))
    ax.bar([x - width / 2 for x in xs], [fnum(r["claude_overall"]) for r in rows],
           width, label="Claude (claude-opus-5)", color=ACCENT)
    ax.bar([x + width / 2 for x in xs], [fnum(r["gemini_overall"]) for r in rows],
           width, label="Gemini (gemini-3.8-flash)", color=ACCENT2)
    for i, r in enumerate(rows):
        if fnum(r["claude_overall"]) != fnum(r["gemini_overall"]):
            ax.annotate("不一致", xy=(i, 5.15), ha="center", fontsize=8, color="crimson")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("総合スコア (1-5)")
    ax.set_ylim(0, 5.8)
    ax.set_title("2審査員による独立採点 — 完全一致 11/12")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "phase3_judges.png", bbox_inches="tight")
    print("生成: phase3_judges.png")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    # 日本語ラベルが使えるフォントを選ぶ
    for cand in ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"]:
        try:
            matplotlib.font_manager.findfont(cand, fallback_to_default=False)
            plt.rcParams["font.family"] = cand
            print(f"フォント: {cand}")
            break
        except Exception:
            continue
    phase1_charts()
    phase2_charts()
    phase3_charts()

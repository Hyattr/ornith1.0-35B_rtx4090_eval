"""Phase 1: システム指標の計測。

入力長を変えながら TTFT / decode速度 / prefill速度 / VRAM を測る。

生成長を揃えるため max_tokens で必ず打ち切る。このとき出る
finish_reason == "length" は意図的なもので、失敗ではない。
"""

import csv
import json
import statistics
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmclient as lm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# 目標入力長。実際の値は応答の prompt_tokens で確認する
INPUT_TOKENS = [256, 1024, 4096, 8192, 16384, 24576]
GEN_TOKENS = 256   # 全条件で生成長を固定
N = 5


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    snap = lm.config_snapshot()
    print("構成:", json.dumps(snap, ensure_ascii=False))

    print("ウォームアップ中...")
    lm.warmup(2)

    rows = []
    for target in INPUT_TOKENS:
        print(f"\n--- 入力 {target} トークン (目標) ---")

        for i in range(N):
            # 毎回 salt を変えてプロンプトの接頭辞を一意にする。
            # これをしないと LM Studio のプロンプトキャッシュがヒットし、
            # TTFT が入力長に依存しない一定値になって prefill が測れない。
            salt = f"{target}-{i}-{uuid.uuid4().hex[:12]}"
            prompt = (lm.filler_prompt(target, salt=salt)
                      + "\n\nSummarise the pattern above in detail.")
            try:
                r = lm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=GEN_TOKENS,
                    tag=f"p1_{target}",
                )
            except lm.LMError as e:
                # sysmem fallback を無効化したので、長い入力では
                # ここでエラーになりうる。失敗ではなく VRAM 上限の発見として記録する。
                print(f"  [{i+1}/{N}] エラー: {str(e)[:120]}")
                rows.append({"target_tokens": target, "run": i + 1, "error": str(e)[:300]})
                break

            v = lm.vram()
            ttft = r["ttft"] or 0
            prefill = (r["prompt_tokens"] / ttft) if ttft else None
            rows.append({
                "target_tokens": target,
                "run": i + 1,
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "reasoning_tokens": r["reasoning_tokens"],
                "ttft_s": round(ttft, 4),
                "decode_tok_s": round(r["tok_per_sec"], 2) if r["tok_per_sec"] else None,
                "prefill_tok_s": round(prefill, 1) if prefill else None,
                "generation_time_s": round(r["generation_time"], 3) if r["generation_time"] else None,
                "wall_time_s": round(r["wall_time"], 3),
                "finish_reason": r["finish_reason"],
                "vram_used_mib": v["vram_used_mib"],
                "vram_free_mib": v["vram_free_mib"],
                "gpu_temp_c": v["gpu_temp_c"],
                "error": "",
            })
            print(f"  [{i+1}/{N}] prompt={r['prompt_tokens']} "
                  f"TTFT={ttft:.3f}s decode={r['tok_per_sec']:.1f}tok/s "
                  f"prefill={prefill:.0f}tok/s VRAM残={v['vram_free_mib']}MiB")

    out = RESULTS / "phase1_systems.csv"
    fields = ["target_tokens", "run", "prompt_tokens", "completion_tokens", "reasoning_tokens",
              "ttft_s", "decode_tok_s", "prefill_tok_s", "generation_time_s", "wall_time_s",
              "finish_reason", "vram_used_mib", "vram_free_mib", "gpu_temp_c", "error"]
    with out.open("w", newline="", encoding="utf-8") as f:
        f.write("# " + json.dumps(snap, ensure_ascii=False) + "\n")
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n保存: {out}")

    print("\n=== 集計 (中央値) ===")
    print(f"{'入力':>8} {'実測prompt':>10} {'TTFT(s)':>9} {'decode':>9} {'prefill':>9} {'VRAM残':>8}")
    summary = []
    for target in INPUT_TOKENS:
        ok = [r for r in rows if r.get("target_tokens") == target and not r.get("error")]
        if not ok:
            print(f"{target:>8} {'— 全て失敗 —':>10}")
            continue
        med = lambda k: statistics.median([r[k] for r in ok if r.get(k) is not None])
        line = {
            "target": target, "prompt_tokens": int(med("prompt_tokens")),
            "ttft_s": round(med("ttft_s"), 3), "decode_tok_s": round(med("decode_tok_s"), 1),
            "prefill_tok_s": round(med("prefill_tok_s"), 0),
            "vram_free_mib": int(med("vram_free_mib")), "n": len(ok),
        }
        summary.append(line)
        print(f"{target:>8} {line['prompt_tokens']:>10} {line['ttft_s']:>9} "
              f"{line['decode_tok_s']:>9} {line['prefill_tok_s']:>9.0f} {line['vram_free_mib']:>8}")

    cache = cache_effect()

    (RESULTS / "phase1_summary.json").write_text(
        json.dumps({"config": snap, "summary": summary, "prompt_cache": cache},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


def cache_effect():
    """プロンプトキャッシュの効果を測る。

    同一プロンプトを2回送り、1回目 (cold) と2回目 (warm) の TTFT を比べる。
    評価スクリプトが同じ入力を繰り返すと2回目以降がキャッシュヒットになり、
    TTFT が実態とかけ離れる。その落とし穴の定量化。
    """
    print("\n=== プロンプトキャッシュの効果 (cold vs warm) ===")
    print(f"{'入力':>8} {'cold TTFT':>11} {'warm TTFT':>11} {'倍率':>8}")
    out = []
    for target in [1024, 4096, 16384]:
        salt = f"cache-{target}-{uuid.uuid4().hex[:12]}"
        prompt = lm.filler_prompt(target, salt=salt) + "\n\nSummarise the pattern above."
        msgs = [{"role": "user", "content": prompt}]
        cold = lm.chat(msgs, max_tokens=64, tag=f"p1cache_cold_{target}")
        warm = lm.chat(msgs, max_tokens=64, tag=f"p1cache_warm_{target}")
        ratio = (cold["ttft"] / warm["ttft"]) if warm["ttft"] else None
        out.append({"target_tokens": target, "prompt_tokens": cold["prompt_tokens"],
                    "cold_ttft_s": round(cold["ttft"], 4), "warm_ttft_s": round(warm["ttft"], 4),
                    "speedup": round(ratio, 1) if ratio else None})
        print(f"{target:>8} {cold['ttft']:>10.3f}s {warm['ttft']:>10.3f}s {ratio:>7.1f}x")

    with (RESULTS / "phase1_prompt_cache.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target_tokens", "prompt_tokens",
                                          "cold_ttft_s", "warm_ttft_s", "speedup"])
        w.writeheader()
        w.writerows(out)
    return out


if __name__ == "__main__":
    run()

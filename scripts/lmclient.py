"""LM Studio 計測用の共通クライアント。

LM Studio 独自の /api/v0/chat/completions を使う。OpenAI 互換の /v1 と違い、
サーバー側で計測した TTFT と tok/s が stats フィールドで返るため、
クライアント側でストリーミングを計時するより正確。

依存は標準ライブラリのみ。
"""

import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = "http://127.0.0.1:1235"
MODEL = "ornith-1.0-35b"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# モデルカードの推奨サンプリング設定
RECOMMENDED = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

# 計測済みの実測値。プロンプト長を目標トークン数に合わせるための概算に使う
CHARS_PER_TOKEN = 2.78


class LMError(RuntimeError):
    pass


def _post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # コンテキスト超過などは本文ではなく HTTP 4xx で返る。
        # 呼び出し側が error キーで判定できるよう、本文を読んで dict に詰め直す。
        try:
            detail = json.loads(e.read().decode("utf-8"))
        except Exception:
            detail = {"error": f"HTTP {e.code} {e.reason}"}
        if "error" not in detail:
            detail = {"error": detail}
        return detail


def _get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(messages, max_tokens=2048, sampling=None, tools=None, tag="run", save=True):
    """1 リクエストを送り、結果を正規化した dict で返す。

    LM Studio は存在しないエンドポイントにも HTTP 200 を返すため、
    ステータスコードではなくボディの error キーで失敗を判定する。
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        **(sampling if sampling is not None else RECOMMENDED),
    }
    if tools:
        payload["tools"] = tools

    t0 = time.perf_counter()
    body = _post("/api/v0/chat/completions", payload)
    wall = time.perf_counter() - t0

    if "error" in body:
        raise LMError(str(body["error"])[:400])

    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%H%M%S_%f")
        (DATA / f"{tag}_{stamp}.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    choice = body["choices"][0]
    msg = choice["message"]
    usage = body.get("usage", {})
    stats = body.get("stats", {})
    details = usage.get("completion_tokens_details", {}) or {}

    content = msg.get("content") or ""
    finish = choice.get("finish_reason")

    return {
        "content": content,
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": finish,
        "stop_reason": stats.get("stop_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "ttft": stats.get("time_to_first_token"),
        "tok_per_sec": stats.get("tokens_per_second"),
        "generation_time": stats.get("generation_time"),
        "wall_time": wall,
        # 空応答には2種類ある。max_tokens 不足による打ち切りだけが失敗で、
        # ツール呼び出しによる空 content は正常。
        "empty_answer": content.strip() == "" and finish == "length",
    }


def filler_prompt(target_tokens, salt=None):
    """目標トークン数に近いダミー入力を作る。

    salt を渡すと先頭に一意な文字列が入り、プロンプト全体の接頭辞が変わる。
    LM Studio は接頭辞が一致するとKVキャッシュを再利用し TTFT が跳ね上がって
    速くなるため、prefill を実測したい場合は必ず salt を変えること。
    salt を省略するとキャッシュに載る（＝ヒット時の挙動を測るのに使う）。

    実際のトークン数は応答の prompt_tokens で確認できるので、
    ここでは概算で足りる。集計では常に実測値を使うこと。
    """
    chars = int(target_tokens * CHARS_PER_TOKEN)
    # キャッシュは接頭辞の一致で効くので、先頭が違えば全体が無効化される。
    # salt を各行に入れるとトークン数が大きく膨張するため、先頭のみに置く。
    head = f"Session {salt}. Unique reference marker for this run.\n" if salt else ""
    lines, total, i = ([head.rstrip("\n")], len(head), 0) if head else ([], 0, 0)
    while total < chars:
        line = f"Line {i:06d}: the quick brown fox jumps over the lazy dog near riverbank {i:06d}."
        lines.append(line)
        total += len(line) + 1
        i += 1
    return "\n".join(lines)


def vram():
    """nvidia-smi から VRAM 使用状況を取得する (MiB)。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()[0]
        used, free, util, temp = [x.strip() for x in out.split(",")]
        return {"vram_used_mib": int(used), "vram_free_mib": int(free),
                "gpu_util_pct": int(util), "gpu_temp_c": int(temp)}
    except Exception:
        return {"vram_used_mib": None, "vram_free_mib": None,
                "gpu_util_pct": None, "gpu_temp_c": None}


def config_snapshot():
    """CSV ヘッダに埋め込む構成情報。数値は構成依存なので必ず一緒に記録する。"""
    snap = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": MODEL}
    try:
        for m in _get("/api/v0/models").get("data", []):
            if m.get("id") == MODEL:
                snap.update({
                    "arch": m.get("arch"),
                    "quantization": m.get("quantization"),
                    "loaded_context": m.get("loaded_context_length"),
                    "max_context": m.get("max_context_length"),
                })
    except Exception:
        pass
    snap.update(vram())
    snap.update(RECOMMENDED)
    return snap


def warmup(n=2):
    """ロード時間やキャッシュ未整備の影響を計測から除くための捨て実行。"""
    for _ in range(n):
        chat([{"role": "user", "content": "Hi"}], max_tokens=64, tag="warmup", save=False)

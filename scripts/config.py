"""APIキーなどの設定を .env から読み込む。

Python は .env を自動では読まない。`os.environ` に載っていない値は
anthropic / google-genai の SDK からも見えないため、ここで明示的に流し込む。

依存を増やさないよう標準ライブラリだけで実装している。python-dotenv を
使いたい場合は load_env() を load_dotenv() に置き換えればよい。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def _force_utf8_output():
    """標準出力を UTF-8 にする。

    Windows のコンソールは既定が cp932 で、絵文字などを print すると
    UnicodeEncodeError でスクリプトごと落ちる。計測の途中で表示の都合だけで
    落ちるのは損失が大きいので、出力側を UTF-8 に寄せる。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # リダイレクト先によっては reconfigure できない


_force_utf8_output()


def load_env(path=ENV_FILE):
    """.env を読んで os.environ に入れる。

    すでに環境変数として設定されている値は上書きしない
    （実際の環境変数が .env より優先される。python-dotenv と同じ挙動）。
    """
    if not path.exists():
        return {}

    loaded = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        # 値を囲むクォートは外す（.env の慣習）
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def require(name, hint=""):
    """必須の設定を取得する。無ければ何をすべきか分かるエラーにする。"""
    value = os.environ.get(name)
    if not value:
        msg = f"{name} が設定されていません。{ROOT / '.env'} に追記してください。"
        if hint:
            msg += f"\n  {hint}"
        raise RuntimeError(msg)
    return value


def status():
    """設定状況を表示する。値そのものは出さない（ログや画面への漏洩防止）。"""
    keys = ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LMSTUDIO_BASE_URL"]
    print(f".env: {'あり' if ENV_FILE.exists() else 'なし'} ({ENV_FILE})")
    for k in keys:
        v = os.environ.get(k)
        if not v:
            print(f"  {k}: 未設定")
        elif k.endswith("_API_KEY"):
            # 先頭数文字だけ表示して、取り違えを判別できるようにする
            print(f"  {k}: 設定あり ({v[:7]}…{v[-4:]}, {len(v)}文字)")
        else:
            print(f"  {k}: {v}")


load_env()


if __name__ == "__main__":
    status()

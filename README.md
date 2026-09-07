# ornith1.0-35B_rtx4090_eval

RTX 4090 一枚（VRAM 24GB）に **Ornith-1.0-35B**（MoE / 4bit量子化）を載せて、
速度・思考トークン・実タスク性能を実測した記録。

計測スクリプト・課題セット・採点ルーブリック・全リクエストの生レスポンス、
そして**失敗した計測のデータも**そのまま残してある。数字だけでは検証できないため。

## 主な結果

| 項目 | 結果 |
|---|---|
| 生成速度 | 137.8 tok/s（入力 25,669 トークン時でも 127.2） |
| 入力処理 | 約 7,000 tok/s で頭打ち |
| ツール呼び出し精度 | 100%（60/60）※天井に張り付き、識別力なし |
| JSON スキーマ準拠 | 100%（60/60）※同上 |
| コード生成 | 2審査員で 4.75 / 4.83（完全一致 91.7%、±1以内 100%） |

⚠️ モデルカードの推奨構成は 8×80GB GPU ノード・context 262,144。
本計測は 24GB 一枚・IQ4_XS 量子化・context 32,768（ネイティブの 1/8）で、
**まったく違う土俵での測定**である点に注意。

## 計測が二度壊れた話

このリポジトリの主眼はモデルの点数ではなく、**計測側の失敗**にある。

1. **プロンプトキャッシュ**を無効化していなかった → TTFT が実態の 1/41 に。
   物理的にありえない値（419,000 tok/s）が出て気づいた
2. **`max_tokens` 不足**で 12問中4問が上限に到達（3問は回答が空）→ 平均 3.58。
   予算を上げ直したら 4.75。エラーは一切出ない

無効になったデータも `*_INVALID_cached.csv` / `*_maxtokens4096.*` として残してある。

## ドキュメント

読む順序:

1. [用語集](docs/glossary.md) — 用語・パラメータの解説と実測値まとめ
2. [Phase 1・2 計測結果](docs/phase1-2-results.md) — 速度・VRAM・思考トークン
3. [Phase 3 計測結果](docs/phase3-results.md) — ツール呼び出し・JSON準拠・コード生成
4. [記事ドラフト](docs/article-draft.md) — note 公開用の原稿
5. [記事ドラフト・基礎編](docs/article-basics-draft.md) — トークン入出力の入門記事（[note 公開版](https://note.com/hyattrt/n/nfb5375d04807)）

## 計測環境

```
モデル      : unsloth/Ornith-1.0-35B-GGUF (UD-IQ4_XS)
アーキテクチャ : qwen35moe (MoE) / context 32,768 (native 262,144)
サーバー     : LM Studio (OpenAI互換API)
ハードウェア  : RTX 4090 24GB / Core i9-14900KF / RAM 64GB
サンプリング  : temperature 0.6 / top_p 0.95 / top_k 20（モデルカード推奨値）
審査員      : claude-opus-5 (Anthropic) / gemini-3.8-flash (Google)
```

量子化形式・コンテキスト長・ハードウェアが違えば数値は変わる。

## 再現手順

```bash
pip install anthropic google-genai jsonschema matplotlib
cp .env.example .env   # ANTHROPIC_API_KEY / GEMINI_API_KEY を記入
```

LM Studio でモデルをロードし、ローカルサーバーを `http://127.0.0.1:1235` で起動する。

```bash
python scripts/phase1_systems.py     # 速度・VRAM・プロンプトキャッシュ
python scripts/phase2_reasoning.py   # max_tokens スイープ・難易度ラダー
python scripts/phase3a_toolcalling.py
python scripts/phase3b_json.py
python scripts/phase3c_coding.py     # 2審査員による採点。API課金が発生する
python scripts/make_charts.py        # results/*.png を生成
```

⚠️ Phase 3C は Claude / Gemini の API を呼ぶため課金が発生する。

生のレスポンスは全て `data/` に保存される。

## ライセンス

[MIT](LICENSE)

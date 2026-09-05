# localllmtest

LM Studio で立てたローカル LLM エンドポイントと Claude API を組み合わせて検証するための実験用リポジトリ。

## 前提

- LM Studio のローカルサーバー: `http://127.0.0.1:1235`（OpenAI 互換 API）
- Claude API キーは `.env` に置く（`.env` は Git 管理外）

## セットアップ

```bash
cp .env.example .env   # ANTHROPIC_API_KEY を記入
```

## ドキュメント

- [用語集](docs/glossary.md) — 本プロジェクトで登場する用語・略語・パラメータの解説と実測値まとめ

- [Phase 1・2 計測結果](docs/phase1-2-results.md) — 速度・VRAM・思考トークンの実測とグラフ

- [Phase 3 計測結果](docs/phase3-results.md) — ツール呼び出し・JSON準拠・コード生成の2審査員評価

- [記事ドラフト](docs/article-draft.md) — note 公開用の原稿

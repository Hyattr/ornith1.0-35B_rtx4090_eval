# localllmtest

LM Studio で立てたローカル LLM エンドポイントと Claude API を組み合わせて検証するための実験用リポジトリ。

## 前提

- LM Studio のローカルサーバー: `http://127.0.0.1:1235`（OpenAI 互換 API）
- Claude API キーは `.env` に置く（`.env` は Git 管理外）

## セットアップ

```bash
cp .env.example .env   # ANTHROPIC_API_KEY を記入
```

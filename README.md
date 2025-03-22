# discord-AI-bot
[![Maintainability](https://api.codeclimate.com/v1/badges/f401ec22fe02106e32e0/maintainability)](https://codeclimate.com/github/tawada/discord-AI-bot/maintainability)

## 概要

discord-AI-botは、Discord上で動作するAIチャットボットです。OpenAI（GPT-4o）、Google Gemini、Anthropic Claude-3のAPIを組み合わせて活用し、柔軟なチャット体験を提供します。

## 主な機能

- 複数のAIモデルをサポート（GPT-4o、Gemini、Claude-3）
- AIモデルの自動フォールバック機能
- 画像解析と要約機能
- URL内容の自動要約
- 外部検索機能（AIの知識が不足している場合）
- カスタマイズ可能なロールプレイ設定

## 必要な環境変数

- `DISCORD_API_KEY`: DiscordのBotトークン
- `OPENAI_API_KEY`: OpenAIのAPIキー
- `GEMINI_API_KEY`: Google GeminiのAPIキー
- `ANTHROPIC_API_KEY`: Anthropic ClaudeのAPIキー
- `CHANNEL_IDS`: ボットが反応するDiscordチャンネルIDのリスト（カンマ区切り）
- `ROLE_PROMPT`: ボットの役割を定義するプロンプト
- `ROLE_NAME`: ボットの表示名

## 任意の環境変数

- `TEXT_MODEL`: 使用するAIモデル（デフォルト: "gemini-2.0-flash"）
  - 利用可能なモデル: "gemini-2.0-flash", "gpt-4o"

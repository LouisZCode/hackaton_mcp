---
title: German Teacher Agent V1
emoji: 📚
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.32.0
app_file: app.py
pinned: false
license: mit
short_description: AI-powered German language learning with multi-agent system
tags:
    - agent-demo-track
    - german-language
    - education
    - multi-agent
    - personalized-learning
---

# German Teacher Agent V1

An AI-powered German language learning system featuring a multi-agent architecture for personalized education.

## Features

🎯 **Multi-Agent System**
- **Gatherer Agent**: Collects student information (name, level, hobbies)
- **German Teacher Agent**: Provides personalized German lessons and grammar feedback

📚 **Personalized Learning**
- Adapts to student level (Beginner, Intermediate, Advanced)
- Uses student hobbies as conversation topics
- Web search integration for current facts about hobbies

🔄 **Learning Flow**
1. **Registration**: Gather student profile information
2. **German Conversation**: 3-turn conversation in German
3. **Grammar Analysis**: AI provides top 3 grammar corrections
4. **Feedback Loop**: Opportunity for focused practice

## Technology Stack

- **LangGraph**: Multi-agent orchestration
- **Anthropic Claude**: Language model with web search
- **Gradio**: Interactive web interface
- **LangFuse**: Observability and tracing
- **CSV Database**: Simple student data storage

## Usage

1. **Registration Tab**: Enter your name, German level, and hobbies
2. **German Class Tab**: Use your Student ID to start personalized lessons
3. **Practice**: Engage in German conversation on your favorite topics
4. **Learn**: Receive grammar feedback and practice suggestions

Built for the Agents-MCP-Hackathon 2025 🏆
<div align="center">
<p align="center">
  <img src="assets/omniagent-logo.png" alt="OmniAgent" width="200">
</p>

# OmniAgent
An agent whose intelligence evolves with every interaction, and whose safety hardens dynamically.

<p align="center">
  <a href="https://yeqing17-2026.github.io/OmniAgent/">Website</a>&nbsp; • &nbsp;
  <a href="https://docs.omniagent.dev">Docs (on the way)</a>&nbsp; • &nbsp;
  <a href="README.md">English</a>&nbsp; • &nbsp;
  <a href="README_CN.md">中文</a>&nbsp;

  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-GPL--3.0-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge" alt="PRs Welcome">
</p>

</div>

**OmniAgent** is an open-source AI Agent framework inspired by OpenClaw. It's the only agent that implements full-dimensional self-evolution (**OmniEvolve**):
- **Skill Self-Evolution**: Through automatic creation, inspection, and repair of skills during interaction, skills evolve in real-time
- **Context Self-Evolution**: Built on a multi-layer information stack architecture, leveraging real-time user interaction feedback and LLM summarization feedback to continuously update memory and user preferences — achieving self-evolving Personalization Context
- **BrainModel Self-Evolution**: Through a novel online reinforcement learning feedback loop, the BrainModel iterates dynamically during interaction

Together, these enable full-dimensional (Skill, Context, BrainModel) self-evolution of the Agent. Additionally, **Hyper Harness** and **Deep Reflexion** modules enhance system safety and task success rate:
- **Hyper Harness**: An efficient, safe, and intelligent execution scaffold that provides systematic support for complex tasks
- **Deep Reflexion**: A dual-layer reflective architecture — real-time risk interception and failure-to-insight conversion — providing a robust guarantee for task success rate

---
**OmniAgent** V.S. **OpenClaw** V.S. **Hermes**

| Dimension | OpenClaw | Hermes | OmniAgent |
| :--- | :--- | :--- | :--- |
| **Skill Evolution** | Static skills, no evolution | **Periodic** post-execution evolution (slow to take effect) | **Real-time** self-evolution during execution (fast to take effect) |
| **Skill Injection** | User Message | User Message | User Message (saves 90% token cost) |
| **Context Evolution** | Static context assembly, no evolution (weak) | Prompt-instruction-based evolution (weak) | Real-time interaction feedback + LLM summarization self-evolution (strong) |
| **BrainModel Evolution** | Fixed model, no evolution | Fixed model, no evolution | Self-deployed model, online RL evolution |
| **Harness Safety** | Static security scanning (bypassable) | Skill trust-level policy, static scanning (bypassable) | **Tool & Skill** trust-level policy + four-layer dynamic security scanning (unbypassable) |
| **Hyper Harness** | None (slow) | None (slow) | Dynamic multi-agent + dynamic concurrent tool execution (fast) |
| **Agent Loop** | ReAct single loop (low success rate) | ReAct single loop (low success rate) | Dual-layer Deep Reflexion loop (high success rate) |


## Core Features

**OmniEvolve (Full-Dimensional Self-Evolution)**: The agent evolves continuously through interaction, and safety hardens dynamically.

- **Proactive Memory**: Based on a multi-layer information stack, a dual-path alignment mechanism of explicit user feedback and implicit LLM induction enables autonomous precipitation and continuous self-evolution of user profiles and memory
- **Skill Self-Evolution**: Through pattern extraction from high-frequency action sequences, skills are natively auto-generated; leveraging dual-path feedback from user interaction and LLM diagnosis, skills are automatically diagnosed and repaired
- **Personalization Context**: Through real-time capture of multi-dimensional preference signals, an adaptive personalized context is constructed, achieving precise alignment between the Agent Loop and individual user preferences
- **BrainModel Self-Evolution**: Through a novel online reinforcement learning (GRPO + PRM) feedback loop, the BrainModel achieves closed-loop self-evolution during interactive use

**Hyper Harness (Super Scaffold)**: A more efficient, safe, and intelligent Harness engine.

- **Progressive Context Loading**: A design pattern inspired by Anthropic Claude Skills — Progressive Disclosure — loading on demand in graduated stages
- **Dynamic Multi-Agent**: Introducing **Sentinel** (planning) and **Guardian** (safety) agents that dynamically analyze task complexity and risk level, activating in real-time to improve success rate and safety
- **Dynamic Concurrent Tool Execution**: Auto-resolves inter-tool dependencies, shifting from serial waiting to async parallel invocation, reducing latency for long-chain tasks
- **Four-Layer Dynamic Security Scanning**: LLM intelligent review → Policy engine → Interactive approval → Execution sandbox. Through trust-level classification, different Skills apply different security policies. Security scanning is unbypassable (industry-first)

**Deep Reflexion (Inner-Outer Dual-Layer Reflective Architecture)**: Improves agent task success rate (PASS@1).

- **Inner-to-Outer Failure Experience Conversion**: Based on LLM-driven automatic root cause analysis (RCA) and heuristic strategy extraction, Reflexion is dynamically injected into the context space, achieving an inner-outer dual-layer collaborative closed-loop reflective correction and intelligent retry
- **Inner Failure Prevention Mechanism**: Three-layer failure prevention system (trajectory repetition, error action repetition, loop pseudo-termination) monitors failure risks and injects context, improving task success rate (PASS@1)

---

## What Can You Do With OmniAgent

| Use Case | What OmniAgent Does |
|----------|-------------------|
| **Workspace & Skills** | Config injection: define Agent personality, tasks, and behavior rules via bootstrap files (AGENTS.md / SOUL.md / CUSTOM.md); Progressive loading: read associated documents in graduated stages (L0/L1/L2) based on conversation depth to prevent Token overflow |
| **Coding & Dev** | Full-lifecycle code handling: write, run, and test code directly in the local environment; Auto-correction: on runtime errors, the Agent reads Traceback and attempts fixes until the program runs |
| **Research & Analysis** | Multi-source web search: auto-invokes search tools and visits multiple pages to extract key information; Knowledge cross-validation: compares information from different sources, outputs a comprehensive report with source annotations |
| **System Admin** | Shell command execution: supports terminal commands in sandbox or host environments; Safety control flow: built-in security scanning system auto-suspends and requests user approval for high-risk commands like delete and format |
| **Multi-Channel** | Unified gateway: manages message routing for Feishu, Discord, Telegram, CLI, and more; Session persistence: seamless switching between clients while maintaining Agent memory consistency |
| **Flexible LLM Backends** | Hybrid model routing: freely combine OpenAI, Claude, DeepSeek, Ollama, and other backends |

---

## Quick Start

**Requirements:** Python 3.11+, an LLM API key (DeepSeek / OpenAI / Anthropic / Ollama / Gemini).

### Installation

```bash
pip install -e .

# Interactive setup — choose provider, enter API key, done
omniagent onboard
```

### Three Ways to Interact

| Mode | Command | Description |
|------|---------|-------------|
| **Terminal** | `omniagent chat` | Interactive chat in your terminal |
| **Web UI** | `omniagent serve` | Start Gateway, open http://127.0.0.1:18790 in your browser |
| **Mobile** (Feishu / Discord / Telegram) | `omniagent serve` | Start Gateway, configure Channel in `config.yaml`, then open a session in your terminal |

---

## Configuration

Configuration is layered: defaults → `~/.omniagent/config.yaml` → environment variables.

```yaml
providers:
  deepseek:
    api_key: "sk-your-key"
    model_id: deepseek-chat

agent:
  model_provider: deepseek
  reflexion_enabled: true
```

> Full configuration reference: [docs.omniagent.dev](https://docs.omniagent.dev) *(coming soon)*

---

## Architecture & Project Structure

```
┌──────────────────────────────────────────────────────┐
│  Channels:  CLI · Web UI · Feishu · Discord · Telegram│
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  Gateway  (WebSocket + HTTP · Session Management)     │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  Reflexion Agent Loop                                │
│                                                      │
│  ┌─────────────────┐  ┌──────────────────────────┐   │
│  │ Deep Reflexion  │  │ Hyper Harness            │   │
│  │ Reflexion Loop  │  │ ┌────────────────────┐   │   │
│  │ Failure Prevent │  │ │Progressive Loading │   │   │
│  └─────────────────┘  │ │Dynamic Tool Exec   │   │   │
│                       │ │4-Layer Sec Scan    │   │   │
│  ┌─────────────────┐  │ │Dynamic Multi-Agent │   │   │
│  │ Sentinel Agent  │  │ └────────────────────┘   │   │
│  │ (Planning)      │  └──────────────────────────┘   │
│  ├─────────────────┤                                 │
│  │ Guardian Agent  │  ┌──────────────────────────┐   │
│  │ (Safety Review) │  │ OmniEvolve               │   │
│  └─────────────────┘  │ ┌────────────────────┐   │   │
│                       │ │Proactive Memory    │   │   │
│                       │ │Skill Self-Evolution│   │   │
│                       │ │Personalization     │   │   │
│                       │ │BrainModel Self-Evo │   │   │
│                       │ └────────────────────┘   │   │
│                       └──────────────────────────┘   │
└──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│  LLM Providers                                       │
│  DeepSeek · OpenAI · Anthropic · Ollama · Gemini     │
│  OpenRouter · vLLM · SGLang · Custom                 │
└──────────────────────────────────────────────────────┘
```

```
omniagent/
├── agents/          # Core: reflexion loop, sentinel, guardian, skill/memory evolution, context management
├── security/        # Policy engine, approval, audit, sandbox
├── tools/           # Built-in tools
├── channels/        # Feishu, Discord, Telegram, Webhook
├── config/          # OmniAgentConfig + sub-configs
├── gateway/         # WebSocket + HTTP server
└── rl/              # GRPO + PRM training pipeline
```

---

## Roadmap

### Near-term

- [ ] **Four-Layer Memory System Self-Evolution (Proactive Memory 2.0)** — Automatically extract and persist long-term memories from conversations, refine the proactive memory system, and resolve conflicts between self-evolution rules
- [ ] **Channel Ecosystem Expansion** — Add WeChat, WeCom, DingTalk, and more connectors; improve channel abstraction layer to reduce integration cost
- [ ] **Plan-Mode** — Implement a new Agent planning mode that generates and confirms a plan before executing complex tasks, then executes step by step
- [ ] **Multi-Agent Collaboration** — Enhance inter-agent communication and task orchestration, support dynamic delegation and result aggregation between agents
- [ ] **Documentation**

---

## Contributing

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Run `python -m pytest` to verify
4. Submit a pull request

We welcome contributions of all kinds — bug fixes, new tools, channel connectors, and documentation improvements.

---

## License

This project is licensed under [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.txt). Any code that references this project must also be open-sourced under the same license.

---

<div align="center">

**OmniAgent** — An agent whose intelligence evolves with every interaction, and whose safety hardens dynamically.

</div>

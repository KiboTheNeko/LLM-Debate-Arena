# LLM Debate Arena

A minimal web application that pits two independently configured LLMs against each other, defending **two topics or positions** on one central question.

## Features

- Two distinct model identities / endpoints (Agent A and Agent B)
- Different OpenAI-compatible backends supported: Ollama local, OpenAI API, OpenRouter, compatible self-hosted gateways
- 1–8 debate rounds; speaker order alternates each round (A→B, then B→A)
- Each turn shows **thesis, claims, supporting rationale, rebuttal, concession, and cross-examination question**
- Side-by-side round view and a searchable/scrollable point comparison table
- Generate a single next response, or complete all remaining rounds
- Resumes from completed turns if a request fails; reset to start fresh
- Download complete Markdown or JSON transcripts
- Clearly labeled template-only demo to preview without API keys
- Optional input/output token counts when provider reports them

## Quick start

Requires Python 3.10+.

```bash
cd llm_debate_arena
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal (normally `http://localhost:8501`).

### Local, free-to-run inference with Ollama

Install Ollama separately, pull a model, and ensure the Ollama daemon is running:

```bash
ollama pull llama3.2
ollama list
```

Select **Live LLM debate**. For both agents choose `Ollama (local)`, URL `http://localhost:11434/v1`, and model `llama3.2`, or configure two different locally installed models. Ollama ignores the dummy key. Local inference consumes RAM/GPU resources, and two backends on the same machine may contend for them.

### Cloud models

Select **OpenAI-compatible API** for an agent, enter the base URL, *exact* model ID, and provider key. Example for OpenAI: `https://api.openai.com/v1`. The two agents may use different providers. API usage is billable under provider pricing; two responses per round means 6 calls for a 3-round debate.

For environment-variable keys:

```bash
cp .env.example .env
# Fill AGENT_A_API_KEY and AGENT_B_API_KEY in .env.
streamlit run app.py
```

The `.env` file is not included in the ZIP. **Never commit API keys or expose this app publicly without access controls.** Keep the app backend private; keys are not included in exported transcripts. When deploying publicly, consider a proper secrets manager and per-user authentication.

## How a debate works

1. Enter a central question, e.g. `Which approach fits a remotely operated UAV platform better?`.
2. Enter side A's position, e.g. `Onboard edge inference`.
3. Enter side B's position, e.g. `Cloud-based inference`.
4. Add evaluation criteria, e.g. `latency, connectivity, power, safety, operating cost`.
5. Configure both models and the round limit.
6. Run the full debate, or generate one response at a time.
7. Review the rounds and export arguments.

The same model can be selected for both sides, but for truly separate LLMs choose distinct model IDs or providers.

## Architecture

```text
Streamlit UI
    │
    ├── Agent A: model+endpoint+key ──→ chat completions
    │
    ├── Agent B: model+endpoint+key ──→ chat completions
    │
    ├── debate_engine.py
    │   ├── alternating scheduler
    │   ├── prompt builder with bounded history
    │   └── JSON parser with plain-text fallback
    │
    └── round cards / point table / Markdown & JSON export
```

Debate participants receive the same question and context, their assigned positions, and a bounded history of previous turns. The engine stores parsed model responses in Streamlit Session State (temporary to the browser session); reloads can clear the debate, so export anything worth retaining.

### Limitations and safeguards

- No independent fact-checking, source retrieval or independent judge yet. Model-provided claims are not validated.
- Models are prompted not to invent sources, but may still hallucinate or agree uncritically.
- Opponent arguments are explicitly treated as untrusted text in the prompt. This reduces but does not eliminate prompt-injection risk.
- Large debates can consume tokens. History is bounded to ~18k characters; older details may fall out of context.
- Chat completions are used to maximize compatibility; model/provider feature support varies. If an endpoint is not compatible, it will not work without an adapter.
- The demo is **static templated content**, never an LLM response.
- Each agent's response is generated sequentially, not in parallel, so the second speaker can rebut the first.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Reference APIs

- OpenAI: https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create
- Ollama: https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx
- Streamlit Session State: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state

"""Side-by-side two-LLM Debate Arena (Streamlit) — Cloud API Edition."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from debate_engine import (
    CLOUD_PROVIDERS,
    demo_response,
    fetch_provider_models,
    live_response,
    markdown_export,
    point_rows,
    schedule,
)

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

st.set_page_config(page_title="LLM Debate Arena", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1350px; padding-top: 2rem; }
.big-sub {color:#8390a1; margin-top:-0.5rem; margin-bottom:1.4rem}
.agent-a {border-top:4px solid #3e9ae7; padding:0.5rem 0.2rem 0}
.agent-b {border-top:4px solid #e7a03e; padding:0.5rem 0.2rem 0}
.small-label {text-transform:uppercase; letter-spacing:0.09em; font-size:.76rem; opacity:.75;}
</style>
""", unsafe_allow_html=True)

for key, value in {"turns": [], "active_id": None, "active_meta": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _default_env_key(side: str, provider_name: str) -> str:
    """Resolve initial API key from environment variables."""
    side_specific = os.getenv(f"AGENT_{side}_API_KEY", "").strip()
    if side_specific:
        return side_specific
    if provider_name == "OpenRouter":
        return os.getenv("OPENROUTER_API_KEY", "").strip()
    if provider_name == "Google AI Studio (Gemini)":
        return (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    return os.getenv("OPENAI_API_KEY", "").strip()


def _init_agent_state(side: str, default_provider: str) -> None:
    prov_key = f"provider_{side}"
    if prov_key not in st.session_state:
        st.session_state[prov_key] = default_provider
    provider_info = CLOUD_PROVIDERS[st.session_state[prov_key]]

    if f"base_url_{side}" not in st.session_state:
        st.session_state[f"base_url_{side}"] = provider_info["base_url"]
    if f"model_id_{side}" not in st.session_state:
        st.session_state[f"model_id_{side}"] = (
            os.getenv(f"AGENT_{side}_MODEL", "").strip() or provider_info["default_model"]
        )
    if f"api_key_{side}" not in st.session_state:
        st.session_state[f"api_key_{side}"] = _default_env_key(side, st.session_state[prov_key])


_init_agent_state("A", "OpenRouter")
_init_agent_state("B", "Google AI Studio (Gemini)")


def _on_provider_change(side: str) -> None:
    provider_name = st.session_state[f"provider_{side}"]
    info = CLOUD_PROVIDERS[provider_name]
    st.session_state[f"base_url_{side}"] = info["base_url"]
    st.session_state[f"model_id_{side}"] = info["default_model"]
    if not st.session_state.get(f"api_key_{side}"):
        env_val = _default_env_key(side, provider_name)
        if env_val:
            st.session_state[f"api_key_{side}"] = env_val


def _on_preset_change(side: str, preset_key: str) -> None:
    chosen = st.session_state.get(preset_key, "")
    if chosen and not chosen.startswith("✏️"):
        st.session_state[f"model_id_{side}"] = chosen


def _save_keys_to_env(config_a: dict[str, str], config_b: dict[str, str]) -> None:
    lines = [
        "# Saved from LLM Debate Arena GUI",
        f"AGENT_A_API_KEY={config_a['api_key']}",
        f"AGENT_B_API_KEY={config_b['api_key']}",
        f"AGENT_A_MODEL={config_a['model']}",
        f"AGENT_B_MODEL={config_b['model']}",
    ]
    for cfg in (config_a, config_b):
        if cfg["type"] == "OpenRouter" and cfg["api_key"]:
            lines.append(f"OPENROUTER_API_KEY={cfg['api_key']}")
        elif cfg["type"] == "Google AI Studio (Gemini)" and cfg["api_key"]:
            lines.append(f"GOOGLE_API_KEY={cfg['api_key']}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


st.title("⚖️ LLM Debate Arena")
st.markdown(
    '<div class="big-sub">Two independent Cloud LLM roles · OpenRouter & Google AI Studio · Alternating rounds · Side-by-side evidence and rebuttals</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Execution Settings")
    mode = st.radio("Mode", ["Live LLM debate", "Demo — no API calls"], index=0)
    is_demo = mode.startswith("Demo")
    rounds = st.slider("Debate rounds", min_value=1, max_value=8, value=3)
    if is_demo:
        st.info("This mode shows **templated example text**, not outputs from LLMs. Switch to Live for real debates.")
    else:
        st.caption("Configure each agent's Cloud API key and model directly in the **Cloud API & Model Configuration** panel on the main page.")


def render_agent_api_panel(side: str, accent_class: str) -> dict[str, str]:
    st.markdown(f'<div class="{accent_class}"></div>', unsafe_allow_html=True)
    st.markdown(f"#### 🤖 Agent {side} · Cloud API & Model")

    provider_names = list(CLOUD_PROVIDERS.keys())
    provider_type = st.selectbox(
        f"Agent {side} Cloud Provider",
        provider_names,
        key=f"provider_{side}",
        on_change=_on_provider_change,
        args=(side,),
    )
    provider_info = CLOUD_PROVIDERS[provider_type]

    key_placeholder = (
        "sk-or-v1-..." if provider_type == "OpenRouter"
        else "AIzaSy..." if provider_type == "Google AI Studio (Gemini)"
        else "sk-..."
    )
    api_key = st.text_input(
        f"Agent {side} · {provider_type} API Key",
        type="password",
        key=f"api_key_{side}",
        placeholder=f"Paste your {provider_type} API key ({key_placeholder})",
        help="Your API key is kept in memory for requests and never included in exported transcripts.",
    )

    fetched_key = f"fetched_models_{side}_{provider_type}"
    fetched_list = st.session_state.get(fetched_key, [])
    preset_options = list(dict.fromkeys(fetched_list + provider_info["preset_models"])) + [
        "✏️ Custom model ID (type below)"
    ]

    preset_widget_key = f"preset_{side}_{provider_type}"
    current_model = st.session_state.get(f"model_id_{side}", provider_info["default_model"])
    default_preset_idx = (
        preset_options.index(current_model)
        if current_model in preset_options
        else len(preset_options) - 1
    )

    pcol, fcol = st.columns([3.2, 1.3])
    with pcol:
        st.selectbox(
            f"Agent {side} · Model Preset / List",
            preset_options,
            index=default_preset_idx,
            key=preset_widget_key,
            on_change=_on_preset_change,
            args=(side, preset_widget_key),
        )
    with fcol:
        st.write("")
        if st.button("🔄 Fetch Models", key=f"fetch_btn_{side}", use_container_width=True,
                     help=f"Fetch available models live from {provider_type}"):
            try:
                models = fetch_provider_models(
                    provider_type,
                    st.session_state.get(f"base_url_{side}", provider_info["base_url"]),
                    api_key,
                )
                if models:
                    st.session_state[fetched_key] = models
                    st.toast(f"Loaded {len(models)} models from {provider_type} for Agent {side}!")
                    st.rerun()
                else:
                    st.warning("No models returned by endpoint.")
            except Exception as exc:
                st.error(f"Could not fetch models: {exc}")

    model = st.text_input(
        f"Agent {side} · Model ID (specify any model)",
        key=f"model_id_{side}",
        placeholder=f"e.g. {provider_info['default_model']}",
        help="Pick from the preset dropdown above or type/paste any exact model ID here.",
    )

    with st.expander(f"Agent {side} · Endpoint URL & Status", expanded=False):
        base_url = st.text_input(
            f"Agent {side} API Base URL",
            key=f"base_url_{side}",
        )

    if api_key.strip():
        st.caption(f"✅ **API key set** · Model: `{model.strip() or '(none)'}`")
    else:
        st.caption(f"🔑 **API key required** for {provider_type} · Model: `{model.strip() or '(none)'}`")

    return {
        "type": provider_type,
        "model": model.strip(),
        "base_url": base_url.strip().rstrip("/"),
        "api_key": api_key.strip(),
    }


st.subheader("☁️ Cloud API & Model Configuration")
api_col_a, api_col_b = st.columns(2, gap="large")
with api_col_a:
    config_a = render_agent_api_panel("A", "agent-a")
with api_col_b:
    config_b = render_agent_api_panel("B", "agent-b")

with st.sidebar:
    st.divider()
    st.subheader("Active Cloud Setup")
    st.markdown(f"**Agent A:** {config_a['type']}  \nModel: `{config_a['model'] or '—'}`  \nKey: {'✅ Set' if config_a['api_key'] else '❌ Missing'}")
    st.markdown(f"**Agent B:** {config_b['type']}  \nModel: `{config_b['model'] or '—'}`  \nKey: {'✅ Set' if config_b['api_key'] else '❌ Missing'}")
    if st.button("💾 Save API Keys & Models to .env", use_container_width=True):
        _save_keys_to_env(config_a, config_b)
        st.success("Saved API keys and models to `.env`!")

st.divider()
st.subheader("Debate brief")
question = st.text_input("Central question", value="Which is preferable for real-time drone AI inference?", max_chars=4000)
col_a, col_b = st.columns(2, gap="large")
with col_a:
    position_a = st.text_area("Agent A · position or topic", value="Onboard edge inference", height=95, max_chars=4000)
with col_b:
    position_b = st.text_area("Agent B · position or topic", value="Cloud-based inference", height=95, max_chars=4000)
context = st.text_area("Shared context / evaluation criteria (optional)",
                       value="Compare latency, connectivity dependence, energy, compute cost, safety, and maintenance.",
                       height=80, max_chars=4000)

brief = {"question": question.strip(), "position_a": position_a.strip(),
         "position_b": position_b.strip(), "context": context.strip()}
models_public = {
    "A": {k: v for k, v in config_a.items() if k != "api_key"},
    "B": {k: v for k, v in config_b.items() if k != "api_key"},
}
metadata = {"brief": brief, "rounds": rounds, "demo": is_demo,
            "models": models_public if not is_demo else {}}
current_id = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

start, next_one, reset = st.columns([1.8, 1.7, 1.0])
with start:
    run_all = st.button("▶ Run / continue all rounds", type="primary", use_container_width=True)
with next_one:
    run_next = st.button("Generate one response", use_container_width=True)
with reset:
    reset_clicked = st.button("Reset debate", use_container_width=True)

if reset_clicked:
    st.session_state.turns = []
    st.session_state.active_id = None
    st.session_state.active_meta = None
    st.rerun()

if st.session_state.active_id and st.session_state.active_id != current_id:
    st.warning("The debate settings have changed. **Reset debate** to apply them; existing rounds remain visible below.")

if run_all or run_next:
    errors = []
    if not brief["question"] or not brief["position_a"] or not brief["position_b"]:
        errors.append("Supply a question and both positions.")
    if brief["position_a"].casefold() == brief["position_b"].casefold():
        errors.append("The two positions must be distinct.")
    if st.session_state.active_id and current_id != st.session_state.active_id:
        errors.append("Reset the existing debate before changing its settings.")
    if not is_demo:
        for side, c in (("A", config_a), ("B", config_b)):
            if not c["model"] or not c["base_url"]:
                errors.append(f"Agent {side} ({c['type']}): specify a Model ID and API Base URL.")
            if not c["api_key"]:
                errors.append(f"Agent {side} ({c['type']}): enter an API key in the GUI above.")
    if errors:
        for error in errors:
            st.error(error)
    else:
        if st.session_state.active_id is None:
            st.session_state.active_id = current_id
            st.session_state.active_meta = metadata
        current_sequence = schedule(rounds)
        target = len(current_sequence) if run_all else min(len(st.session_state.turns) + 1, len(current_sequence))
        for ix in range(len(st.session_state.turns), target):
            round_number, side = current_sequence[ix]
            side_cfg = config_a if side == "A" else config_b
            st.write(f"Generating **round {round_number} · Agent {side}** ({side_cfg['type']} · `{side_cfg['model']}`) …")
            try:
                reply = (
                    demo_response(brief, side, round_number, st.session_state.turns)
                    if is_demo else
                    live_response(brief, side, round_number, st.session_state.turns, side_cfg)
                )
                st.session_state.turns.append({
                    "round": round_number, "side": side,
                    "model": "Template demo" if is_demo else side_cfg["model"],
                    **reply,
                })
            except Exception as exc:
                st.error(f"Agent {side} ({side_cfg['type']}) failed in round {round_number}: {type(exc).__name__}: {exc}")
                st.info("Completed responses are retained. Check your API key or model ID and click Run / continue to resume.")
                break

turns = st.session_state.turns
active_meta = st.session_state.active_meta or metadata
active_brief = active_meta["brief"]
active_rounds = active_meta["rounds"]
st.divider()

if not turns:
    st.info("Enter your **OpenRouter** and **Google AI Studio** API keys above, choose or type your models, and click **▶ Run / continue all rounds** to start the debate.")
else:
    total = len(schedule(active_rounds))
    st.progress(len(turns) / total, text=f"{len(turns)} / {total} responses generated")
    if active_meta["demo"]:
        st.warning("**Template demo only:** the statements below are prewritten UI examples, not LLM-generated arguments.")
    else:
        st.caption("Model responses are arguments, not independently verified facts. Verify numerical and technical claims before use.")

    st.subheader("Round-by-round debate")
    turn_lookup = {(t["round"], t["side"]): t for t in turns}
    for round_number in range(1, active_rounds + 1):
        if all((round_number, side) not in turn_lookup for side in ("A", "B")):
            continue
        st.markdown(f"#### Round {round_number}")
        left, right = st.columns(2, gap="large")
        for side, pane in (("A", left), ("B", right)):
            turn = turn_lookup.get((round_number, side))
            with pane:
                st.markdown(f'<div class="agent-{side.lower()}"></div>', unsafe_allow_html=True)
                model_badge = f" · `{turn.get('model')}`" if turn and turn.get("model") else ""
                st.markdown(f"**Agent {side} — {active_brief[f'position_{side.lower()}']}**{model_badge}")
                if turn is None:
                    st.caption("Awaiting response")
                    continue
                arg = turn["argument"]
                st.write(arg.get("thesis") or "(No thesis)")
                for i, point in enumerate(arg.get("points", []), start=1):
                    st.markdown(f"**{i}. {point['claim']}**")
                    if point.get("support"):
                        st.caption(point["support"])
                if arg.get("rebuttal"):
                    with st.expander("↳ Rebuttal", expanded=True):
                        st.write(arg["rebuttal"])
                if arg.get("concession"):
                    with st.expander("Concession / uncertainty"):
                        st.write(arg["concession"])
                if arg.get("question"):
                    st.markdown(f"*Question to opponent:* {arg['question']}")
                usage = turn.get("usage") or {}
                if usage.get("input_tokens") is not None:
                    st.caption(f"Tokens: {usage.get('input_tokens', '?')} input · {usage.get('output_tokens', '?')} output")
        st.divider()

    st.subheader("All arguments · comparison table")
    rows = point_rows(turns)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)

    transcript = {
        "question": active_brief["question"],
        "position_a": active_brief["position_a"],
        "position_b": active_brief["position_b"],
        "context": active_brief.get("context", ""),
        "demo_not_llm_generated": active_meta["demo"],
        "models": active_meta.get("models", {}),
        "turns": turns,
    }
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button("↓ Download Markdown transcript",
             markdown_export(active_brief, turns, active_meta["demo"]),
             file_name="debate_transcript.md", mime="text/markdown", use_container_width=True)
    with dcol2:
        st.download_button("↓ Download JSON transcript",
             json.dumps(transcript, indent=2, ensure_ascii=False),
             file_name="debate_transcript.json", mime="application/json", use_container_width=True)

st.caption("Designed for debates and comparative trade-offs. A topic does not become a fact simply because an LLM argues for it.")

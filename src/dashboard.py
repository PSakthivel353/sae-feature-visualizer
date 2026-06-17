"""
dashboard.py — SAE Feature Visualizer: Interactive Interpretability Dashboard
─────────────────────────────────────────────────────────────────────────────
A Streamlit application for exploring trained Sparse Autoencoder features.

FEATURES:
  • Checkpoint selector       — switch between trained SAE checkpoints
  • Feature selector          — browse all 4096 features by ID
  • Feature search            — search features by keyword in top examples
  • Activation histogram      — distribution of a feature's activation values
  • Top examples table        — highest-activating tokens/sentences
  • Feature statistics        — frequency, mean/max activation, dead-feature flag
  • Live inference panel      — type a sentence, see which features fire
  • Plotly visualizations throughout

RUN:
  streamlit run src/dashboard.py
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from sae_model import SparseAutoencoder

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SAE Feature Visualizer",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0B0E14;
    --bg-secondary: #11151D;
    --bg-card: #161B26;
    --border-subtle: #232A38;
    --accent: #5EEAD4;
    --accent-dim: #2DD4BF;
    --accent-warn: #FB923C;
    --text-primary: #E5E9F0;
    --text-secondary: #8B95A7;
    --text-mono: #C5D1E8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: var(--bg-primary);
}

/* Headers */
h1, h2, h3 {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
}

code, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Top banner */
.lab-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}
.lab-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
}
.lab-title .accent {
    color: var(--accent);
}
.lab-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--accent);
}
.metric-value.warn { color: var(--accent-warn); }

/* Concept explainer boxes */
.concept-box {
    background: var(--bg-secondary);
    border-left: 3px solid var(--accent-dim);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--text-primary);
}
.concept-box .concept-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--accent-dim);
    margin-bottom: 0.5rem;
    display: block;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

/* Badge */
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.15rem 0.55rem;
    border-radius: 3px;
    letter-spacing: 0.03em;
}
.badge-live { background: rgba(94,234,212,0.12); color: var(--accent); border: 1px solid rgba(94,234,212,0.3); }
.badge-dead { background: rgba(251,146,60,0.12); color: var(--accent-warn); border: 1px solid rgba(251,146,60,0.3); }

hr { border-color: var(--border-subtle); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS / CACHED LOADERS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def discover_checkpoints(checkpoint_dir: str = "checkpoints"):
    """Find all .pt checkpoint files that look like SAE weights."""
    path = Path(checkpoint_dir)
    if not path.exists():
        return []
    files = sorted(path.glob("*.pt"))
    # Exclude known non-model files
    excluded = {"normalization_stats.pt"}
    return [f for f in files if f.name not in excluded]


@st.cache_resource
def load_sae_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        d_model = ckpt.get("d_model", 768)
        d_hidden = ckpt.get("d_hidden", 4096)
        sae = SparseAutoencoder(d_model=d_model, d_hidden=d_hidden)
        sae.load_state_dict(ckpt["model_state_dict"])
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    else:
        sae = SparseAutoencoder()
        sae.load_state_dict(ckpt)
        meta = {}
    sae.eval()
    return sae, meta


@st.cache_data
def load_feature_examples(path: str = "cache/feature_top_examples.json"):
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_feature_stats(path: str = "cache/feature_stats.json"):
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_gpt2_for_inference():
    from transformers import GPT2Model, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_normalization_stats(path: str = "checkpoints/normalization_stats.pt"):
    if not Path(path).exists():
        return None
    return torch.load(path, map_location="cpu")


def search_features(examples: dict, query: str, max_results: int = 30):
    """Search features by keyword match in their top-activating sentences."""
    query = query.lower().strip()
    if not query:
        return []
    matches = []
    for fid, exs in examples.items():
        text_blob = " ".join(e.get("sentence", "") + " " + e.get("token", "") for e in exs).lower()
        if query in text_blob:
            matches.append(int(fid))
        if len(matches) >= max_results:
            break
    return sorted(matches)


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="lab-header">
        <div>
            <div class="lab-title">◆ SAE <span class="accent">Feature</span> Visualizer</div>
        </div>
        <div class="lab-subtitle">Mechanistic Interpretability · GPT-2 · Layer Inspector</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — CHECKPOINT SELECTOR + NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("##### ⚙ Model Checkpoint")
    checkpoints = discover_checkpoints()

    if not checkpoints:
        st.warning(
            "No checkpoints found in `checkpoints/`.\n\n"
            "Train one first:\n```\npython src/train_sae.py\n```"
        )
        st.stop()

    checkpoint_names = [c.name for c in checkpoints]
    selected_name = st.selectbox(
        "Checkpoint file",
        checkpoint_names,
        index=len(checkpoint_names) - 1,
        help="Switch between SAE training runs (e.g. before/after lambda tuning)",
    )
    selected_path = str(Path("checkpoints") / selected_name)

    sae, ckpt_meta = load_sae_checkpoint(selected_path)

    if ckpt_meta:
        st.caption(
            f"d_hidden: `{sae.d_hidden}` · "
            f"λ: `{ckpt_meta.get('l1_lambda', '—')}` · "
            f"epoch: `{ckpt_meta.get('epoch', '—')}`"
        )

    st.divider()
    st.markdown("##### 🔬 Layer")
    layer = st.slider("GPT-2 layer hooked", 0, 11, 8, help="Must match the layer used during activation collection / training")

    st.divider()
    page = st.radio(
        "Navigate",
        ["Feature Explorer", "Live Inference", "About This Project"],
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("📖 What am I looking at?", expanded=False):
        st.markdown(
            """
            <div class="concept-box">
            <span class="concept-title">Hidden States</span>
            Every time GPT-2 processes a token, each layer produces a vector
            (768 numbers) called a <b>hidden state</b> — its internal,
            mid-computation "thought" about that token in context. These
            vectors are usually opaque: individual dimensions rarely mean
            anything on their own.
            </div>
            <div class="concept-box">
            <span class="concept-title">Sparse Autoencoder</span>
            An SAE is a small neural net trained to reconstruct hidden states
            through a much wider, sparsely-active middle layer. Because
            only ~20–50 of 4096 "features" fire per token, each feature is
            pushed toward representing one clean, human-interpretable concept.
            </div>
            <div class="concept-box">
            <span class="concept-title">Feature Discovery</span>
            We discover what a feature means empirically: collect its
            top-activating tokens across a large corpus and read them. If
            they share an obvious theme (legal language, negation, code
            syntax), the feature has a discoverable, "monosemantic" meaning.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

if page == "Feature Explorer":

    examples = load_feature_examples()
    stats = load_feature_stats()

    if not examples or not stats:
        st.info(
            "No feature analysis data found. Run:\n\n"
            "```\npython src/feature_analysis.py --checkpoint checkpoints/"
            + selected_name + "\n```\n\nto generate `feature_top_examples.json` and `feature_stats.json`."
        )
        st.stop()

    # ── Top-level corpus stats row ────────────────────────────────────────
    n_features = len(stats)
    n_dead = sum(1 for s in stats.values() if s.get("is_dead"))
    n_live = n_features - n_dead
    avg_freq = np.mean([s["frequency"] for s in stats.values()]) if stats else 0

    col1, col2, col3, col4 = st.columns(4)
    for col, label, value, warn in [
        (col1, "TOTAL FEATURES", f"{n_features:,}", False),
        (col2, "LIVE FEATURES", f"{n_live:,}", False),
        (col3, "DEAD FEATURES", f"{n_dead:,}", n_dead / max(n_features,1) > 0.3),
        (col4, "AVG FREQUENCY", f"{avg_freq*100:.2f}%", False),
    ]:
        with col:
            cls = "metric-value warn" if warn else "metric-value"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="{cls}">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature selection: search OR direct ID ────────────────────────────
    search_col, select_col = st.columns([2, 1])

    with search_col:
        search_query = st.text_input(
            "🔎 Search features by keyword",
            placeholder="e.g. legal, negation, French, code, biology...",
            help="Searches the top-activating example sentences for every feature",
        )

    with select_col:
        all_ids = sorted(int(k) for k in stats.keys())
        if search_query:
            search_results = search_features(examples, search_query)
            options = search_results if search_results else all_ids
            if not search_results:
                st.caption(f"No matches for '{search_query}' — showing all features")
        else:
            options = all_ids

        feature_id = st.selectbox(
            "Feature ID",
            options,
            format_func=lambda x: f"Feature {x}" + (" (dead)" if stats.get(str(x), {}).get("is_dead") else ""),
        )

    if search_query:
        results = search_features(examples, search_query)
        st.caption(f"Found **{len(results)}** matching features for '{search_query}'")

    st.divider()

    # ── Selected feature deep-dive ─────────────────────────────────────────
    fstats = stats.get(str(feature_id), {})
    fexamples = examples.get(str(feature_id), [])

    is_dead = fstats.get("is_dead", False)
    badge = '<span class="badge badge-dead">DEAD FEATURE</span>' if is_dead else '<span class="badge badge-live">ACTIVE</span>'

    st.markdown(f"### Feature `{feature_id}` &nbsp; {badge}", unsafe_allow_html=True)

    # Feature statistics row
    stat_cols = st.columns(4)
    stat_items = [
        ("FREQUENCY", f"{fstats.get('frequency', 0)*100:.3f}%"),
        ("MEAN ACTIVATION", f"{fstats.get('mean_activation', 0):.3f}"),
        ("MAX ACTIVATION", f"{fstats.get('max_activation', 0):.3f}"),
        ("TOP EXAMPLES FOUND", f"{len(fexamples)}"),
    ]
    for col, (label, val) in zip(stat_cols, stat_items):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 ACTIVATION HISTOGRAM", "📋 TOP EXAMPLES"])

    with tab1:
        if fexamples:
            scores = [e["score"] for e in fexamples]
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=scores,
                nbinsx=20,
                marker_color="#5EEAD4",
                marker_line_color="#0B0E14",
                marker_line_width=1,
                opacity=0.85,
            ))
            fig.update_layout(
                title=f"Top-{len(scores)} Activation Score Distribution — Feature {feature_id}",
                xaxis_title="Activation strength",
                yaxis_title="Count",
                template="plotly_dark",
                plot_bgcolor="#161B26",
                paper_bgcolor="#161B26",
                font=dict(family="Inter", color="#E5E9F0"),
                height=380,
                margin=dict(t=60, l=10, r=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This feature has no recorded activations (dead feature) — nothing to plot.")

    with tab2:
        if fexamples:
            df = pd.DataFrame(fexamples)
            df.index = df.index + 1
            df.index.name = "Rank"
            df = df.rename(columns={"token": "Token", "sentence": "Source Sentence", "score": "Activation"})
            st.dataframe(
                df[["Token", "Source Sentence", "Activation"]],
                use_container_width=True,
                height=420,
                column_config={
                    "Activation": st.column_config.ProgressColumn(
                        "Activation",
                        min_value=0,
                        max_value=max(df["Activation"]) if len(df) else 1,
                        format="%.3f",
                    ),
                },
            )
        else:
            st.info("No top examples recorded — this feature never activated across the analyzed corpus.")

    # ── Cross-feature overview: frequency distribution ─────────────────────
    st.divider()
    st.markdown("##### Corpus-Wide Feature Frequency Distribution")
    st.caption("Where does this feature sit relative to all others? Most features should be rare and specific.")

    freqs = sorted([s["frequency"] for s in stats.values()], reverse=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=freqs, mode="lines", fill="tozeroy",
        line_color="#2DD4BF", fillcolor="rgba(45,212,191,0.1)",
        name="All features",
    ))
    this_rank = sorted([s["frequency"] for s in stats.values()], reverse=True).index(fstats.get("frequency", 0)) if fstats else None
    if this_rank is not None:
        fig2.add_trace(go.Scatter(
            x=[this_rank], y=[fstats.get("frequency", 0)],
            mode="markers", marker=dict(size=12, color="#FB923C", symbol="diamond"),
            name=f"Feature {feature_id}",
        ))
    fig2.update_layout(
        xaxis_title="Feature rank (sorted by frequency)",
        yaxis_title="Activation frequency",
        template="plotly_dark",
        plot_bgcolor="#161B26",
        paper_bgcolor="#161B26",
        font=dict(family="Inter", color="#E5E9F0"),
        height=320,
        margin=dict(t=20, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: LIVE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Live Inference":
    st.markdown("### Live Inference")
    st.markdown(
        '<div class="concept-box"><span class="concept-title">How this works</span>'
        "Type any sentence below. We run it through GPT-2, capture the hidden "
        "state at your selected layer for each token, then pass it through the "
        "SAE encoder to see which of the 4096 features activate most strongly. "
        "This is exactly how the corpus-wide feature catalogue was built — just "
        "for a single sentence, live.</div>",
        unsafe_allow_html=True,
    )

    norm_stats = load_normalization_stats()
    if norm_stats is None:
        st.warning(
            "No `checkpoints/normalization_stats.pt` found. Activations will be "
            "used unnormalized, which will not match the SAE's training distribution. "
            "Re-run `train_sae.py` to generate this file."
        )

    user_text = st.text_input(
        "Enter a sentence:",
        value="The lawyer argued brilliantly in front of the judge.",
        help="Try contrasting domains, e.g. legal vs. biological vs. casual language",
    )

    if user_text:
        with st.spinner("Running GPT-2 forward pass..."):
            model, tokenizer = load_gpt2_for_inference()
            from hooks import attach_hooks, get_activation, remove_hooks

            tokens = tokenizer(user_text, return_tensors="pt")
            token_strings = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

            attach_hooks(model, layers=[layer])
            with torch.no_grad():
                model(**tokens)
            act = get_activation(f"layer_{layer}")  # (1, seq_len, 768)
            remove_hooks()

        selected_token = st.selectbox("Inspect token:", token_strings, index=min(2, len(token_strings)-1))
        tok_idx = token_strings.index(selected_token)

        vec = act[0, tok_idx, :].unsqueeze(0).float()
        if norm_stats is not None:
            vec = (vec - norm_stats["mean"]) / (norm_stats["std"] + 1e-8)

        with torch.no_grad():
            hidden = sae.encode(vec)

        k = min(15, sae.d_hidden)
        top_features = hidden[0].topk(k)
        active_count = (hidden[0] > 0).sum().item()

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">TOKEN</div><div class="metric-value">"{selected_token}"</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">ACTIVE FEATURES</div><div class="metric-value">{active_count}</div></div>', unsafe_allow_html=True)
        with m3:
            health = "✅ Healthy" if 20 <= active_count <= 50 else "⚠️ Outside target"
            st.markdown(f'<div class="metric-card"><div class="metric-label">SPARSITY HEALTH</div><div class="metric-value" style="font-size:1.1rem">{health}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"##### Top {k} Features for Token `{selected_token}`")

        fig = go.Figure(go.Bar(
            x=top_features.values.tolist(),
            y=[f"Feature {i}" for i in top_features.indices.tolist()],
            orientation="h",
            marker_color="#5EEAD4",
        ))
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#161B26",
            paper_bgcolor="#161B26",
            font=dict(family="Inter", color="#E5E9F0"),
            height=420,
            xaxis_title="Activation strength",
            margin=dict(t=20, l=10, r=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "💡 Tip: look up these feature IDs in the **Feature Explorer** tab to "
            "see their corpus-wide top examples and infer what concept they encode."
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT THIS PROJECT
# ═══════════════════════════════════════════════════════════════════════════

elif page == "About This Project":
    st.markdown("### About This Project")

    st.markdown(
        """
        <div class="concept-box">
        <span class="concept-title">Hidden States — what GPT-2 is "thinking"</span>
        A transformer like GPT-2 processes text token by token, and at every one
        of its 12 layers it produces a <b>hidden state</b>: a vector of 768
        numbers representing that token's evolving meaning in context. By the
        time it reaches the final layer, this vector contains everything the
        model "knows" about that token at that point in the sentence. The
        problem: these 768 numbers are <b>entangled</b> — individual dimensions
        almost never correspond to one clean human concept. This phenomenon is
        called <b>superposition</b>: the model is packing more concepts than it
        has dimensions, by overlapping them.
        </div>

        <div class="concept-box">
        <span class="concept-title">Sparse Autoencoders — untangling superposition</span>
        A Sparse Autoencoder (SAE) is trained to reconstruct these hidden states
        through a much <b>wider</b> intermediate layer (4096 features vs. 768
        input dimensions) while being penalized for using too many features at
        once (an L1 sparsity penalty on the hidden layer). The result: instead
        of 768 entangled dimensions, you get 4096 candidate features where only
        ~20–50 fire for any given token — and each one tends to correspond to a
        single, nameable concept (a legal term, a French loanword, a negation,
        a code-syntax token, and so on).
        </div>

        <div class="concept-box">
        <span class="concept-title">Feature discovery — reading the model's mind empirically</span>
        We don't assign meanings to features by inspecting weights directly —
        that's intractable at this scale. Instead we discover meaning
        empirically: run the SAE across a large text corpus, and for every
        feature, record the tokens that activate it most strongly. If those
        top examples share an obvious theme, the feature is interpretable.
        This dashboard is built entirely around that workflow: browse, search,
        and inspect the discovered features.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("##### Pipeline")
    st.code(
        """
GPT-2 (frozen)              Sparse Autoencoder            Feature Catalogue
┌─────────────┐   hook    ┌──────────────────┐  corpus  ┌──────────────────┐
│ 12 layers   │ ────────▶ │ 768 → 4096 → 768  │ ───────▶ │ top examples per │
│ d_model=768 │  layer 8  │ ReLU + L1 penalty  │  sweep   │ feature, stats    │
└─────────────┘           └──────────────────┘          └──────────────────┘
        """,
        language="text",
    )

    st.markdown("##### Research Lineage")
    st.markdown(
        "This project follows the same methodology as published interpretability "
        "research — most notably Anthropic's work decomposing transformer activations "
        "into monosemantic features using sparse dictionary learning. Building even a "
        "small-scale GPT-2 version exercises the same core skills: hooking model "
        "internals, training a dictionary-learning model, and validating that "
        "discovered features are semantically meaningful."
    )

    st.markdown("##### Resources")
    st.markdown(
        "- Source code: `src/hooks.py`, `src/collect_activations.py`, `src/sae_model.py`, "
        "`src/train_sae.py`, `src/feature_analysis.py`, `src/dashboard.py`\n"
        "- Full documentation: see `README.md`"
    )

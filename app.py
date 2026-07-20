"""
app.py
Streamlit UI for the Voice Assistant (English + Urdu customer support) --
sidebar navigation, chat interface with a mic recorder, language panel
with flag tags, colored quick-action shortcuts, live stats, and recent
history. Opens directly to the Dashboard -- no login step.

Note on HTML usage: every card/section below uses Streamlit's native
st.container(border=True) instead of hand-written <div> wrappers. Manually
opening a <div> in one st.markdown() call, adding widgets, then closing it
in a later st.markdown() call does NOT nest the widgets inside that div --
each st.markdown() call is its own independent element in the page, so the
tags end up unmatched/floating. Any HTML below is only used for small,
fully self-contained snippets (opened and closed within the same string).

Run with:
    streamlit run app.py
"""

import os
import time
import datetime

import streamlit as st
import pandas as pd
import plotly.express as px

from model import load_data, preprocess_data, train_intent_model
from logic import (
    transcribe_audio,
    detect_language,
    run_model_or_algorithm,
    generate_explanation,
    generate_response,
    synthesize_speech,
    language_display_name,
    language_flag,
    LANGUAGE_NAMES,
    SPEECH_AVAILABLE,
    TRANSLATOR_AVAILABLE,
)
from styles import CUSTOM_CSS

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "intents.csv")

st.set_page_config(
    page_title="Voice Assistant | English & Urdu Customer Support",
    page_icon="🎙️",
    layout="wide",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "page": "Dashboard",
        "chat_history": [],   # list of dicts: role, text, lang, intent, time, explanation
        "response_times": [],
        "pending_query": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------------------------------------------------------------------------
# Cached model training (Problem Setup + Core Logic depend on this)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training intent classification model...")
def get_trained_models():
    raw = load_data(DATA_PATH)
    data = preprocess_data(raw)
    clf_lr, vec_lr, metrics_lr = train_intent_model(data, model_type="logreg")
    clf_nb, vec_nb, metrics_nb = train_intent_model(data, model_type="naive_bayes")
    return {
        "logreg": (clf_lr, vec_lr, metrics_lr),
        "naive_bayes": (clf_nb, vec_nb, metrics_nb),
    }, data


models, dataset = get_trained_models()
clf, vectorizer, metrics = models["logreg"]


# ---------------------------------------------------------------------------
# Shared: process one query through the full pipeline
# ---------------------------------------------------------------------------
def process_query(query_text):
    start = time.perf_counter()
    language = detect_language(query_text)
    intent, confidence, all_probs = run_model_or_algorithm(query_text, clf, vectorizer)
    explanation = generate_explanation(query_text, intent, vectorizer, clf)
    response_text, was_translated = generate_response(intent, language)
    elapsed = time.perf_counter() - start
    st.session_state.response_times.append(elapsed)

    now = datetime.datetime.now().strftime("%I:%M %p")

    st.session_state.chat_history.append({
        "role": "user", "text": query_text, "lang": language, "time": now,
    })
    st.session_state.chat_history.append({
        "role": "assistant", "text": response_text, "lang": language,
        "intent": intent, "confidence": confidence, "explanation": explanation,
        "probs": all_probs, "time": now, "translated": was_translated and language == "ur",
    })


# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------
def render_topbar():
    c1, c2, c3, c4 = st.columns([3, 1.3, 0.4, 0.4])
    with c1:
        st.markdown(
            """
            <div class="va-brand">
                <div class="va-brand-icon">🤖</div>
                <div>
                    <p class="va-brand-title">Voice Assistant</p>
                    <p class="va-brand-sub">English &amp; Urdu Customer Support</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        options = ["Auto Detect"] + list(LANGUAGE_NAMES.values())
        st.selectbox("Language", options, key="lang_override", label_visibility="collapsed")
    with c3:
        if st.button("☀️", key="theme_toggle", help="Theme settings"):
            st.session_state.page = "Settings"
            st.rerun()
    with c4:
        st.markdown(
            """
            <div class="va-avatar-wrap">
                <div class="va-topbar-avatar">🧑‍💻</div>
                <div class="va-online-dot"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()


# ---------------------------------------------------------------------------
# Chat bubble (fully self-contained HTML -- opened and closed in one call)
# ---------------------------------------------------------------------------
def render_chat_bubble(msg, idx):
    """
    Renders one chat message using Streamlit's native st.chat_message
    component rather than hand-written HTML. An earlier version built
    each bubble as one big multi-line HTML string; when an optional piece
    (like the "translated" note) was empty, it left a whitespace-only
    line in the middle of that HTML block. Markdown treats a blank line
    as a block separator, so everything after it got parsed as a *new*
    block -- and because it was still indented, that remainder rendered
    as a literal code block instead of live HTML (the raw "<div>...”
    text bug). st.chat_message avoids this whole failure mode entirely.
    """
    is_user = msg["role"] == "user"
    avatar = "🧑" if is_user else "🤖"

    with st.chat_message("user" if is_user else "assistant", avatar=avatar):
        st.write(msg["text"])

        if is_user:
            flag = language_flag(msg["lang"])
            st.caption(
                f"Detected Language: {language_display_name(msg['lang'])} {flag}  ·  {msg['time']}"
            )
        else:
            meta_line = msg["time"]
            if msg.get("translated"):
                meta_line = f"_(Translated from English)_  ·  {meta_line}"
            st.caption(meta_line)

            try:
                audio_out = synthesize_speech(msg["text"], msg["lang"])
                st.audio(audio_out)
            except Exception:
                st.caption("🔇 Speech playback unavailable (needs internet access)")

            if msg.get("explanation"):
                with st.expander("Why this answer? (explainability + confidence)"):
                    st.write(msg["explanation"])
                    prob_df = pd.DataFrame({
                        "Intent": list(msg["probs"].keys()),
                        "Confidence": list(msg["probs"].values()),
                    }).sort_values("Confidence", ascending=False)
                    fig = px.bar(
                        prob_df, x="Intent", y="Confidence", color="Confidence",
                        color_continuous_scale="Purples",
                    )
                    fig.update_layout(yaxis_tickformat=".0%", height=260, margin=dict(t=10, b=10))
                    # Unique key per message index -- without this, every
                    # assistant reply's chart shares the same auto-generated
                    # ID and Streamlit raises StreamlitDuplicateElementId.
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")


# ---------------------------------------------------------------------------
# DASHBOARD / VOICE ASSISTANT PAGE
# ---------------------------------------------------------------------------
def render_dashboard():
    main_col, side_col = st.columns([2.3, 1], gap="medium")

    with main_col:
        # --- A) Problem Setup Module: mic + text input, with validation ---
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center; padding:0.6rem 0 0.2rem 0;">
                    <div class="va-mic-circle">🎤</div>
                    <div class="va-mic-caption">Click the microphone and start speaking</div>
                    <div class="va-mic-status">I'm listening...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            audio_value = st.audio_input(
                "Record your query", label_visibility="collapsed", key="mic_input"
            )

        if audio_value is not None:
            if not SPEECH_AVAILABLE:
                st.error("speech_recognition is not installed in this environment.")
            else:
                temp_path = "temp_input.wav"
                with open(temp_path, "wb") as f:
                    f.write(audio_value.getbuffer())
                try:
                    with st.spinner("Transcribing..."):
                        text = transcribe_audio(temp_path)
                except RuntimeError as e:
                    st.error(f"Transcription error: {e}")
                    text = None
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                if text is None:
                    st.error("❌ Could not understand the audio. Please try again or type below.")
                elif text.strip() == "":
                    st.warning("⚠️ Empty transcript detected. Please try again.")
                else:
                    process_query(text.strip())

        # --- Chat history area ---
        with st.container(border=True, height=440):
            if not st.session_state.chat_history:
                st.caption("No messages yet — record your voice or type a message below to start.")
            for i, msg in enumerate(st.session_state.chat_history):
                render_chat_bubble(msg, i)

        # --- Bottom input row: text field + send + mic shortcut ---
        ic1, ic2, ic3 = st.columns([8, 1, 1])
        with ic1:
            typed = st.text_input(
                "Message", placeholder="Type your message here...",
                label_visibility="collapsed", key="typed_message",
            )
        with ic2:
            send_clicked = st.button("➤", use_container_width=True, key="send_btn")
        with ic3:
            mic_shortcut = st.button("🎤", use_container_width=True, key="mic_shortcut_btn")

        if mic_shortcut:
            st.info("🎙️ Use the microphone panel above to record your voice.")

        if send_clicked:
            if not typed or typed.strip() == "":
                st.warning("⚠️ Please enter a non-empty message.")
            else:
                process_query(typed.strip())
                st.rerun()

        if st.session_state.pending_query:
            process_query(st.session_state.pending_query)
            st.session_state.pending_query = None
            st.rerun()

    with side_col:
        # --- Language card ---
        with st.container(border=True):
            st.markdown('<div class="va-section-title">🌐 Language</div>', unsafe_allow_html=True)
            last_lang = None
            for m in reversed(st.session_state.chat_history):
                if m["role"] == "user":
                    last_lang = m["lang"]
                    break
            detected_str = (
                f"{language_display_name(last_lang)} {language_flag(last_lang)}"
                if last_lang else "—"
            )
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:0.4rem;'>"
                f"<span class='va-dot-green'></span>"
                f"<b>Detected Language: {detected_str}</b></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Supported: " + ", ".join(f"{language_flag(c)} {n}" for c, n in LANGUAGE_NAMES.items())
            )
            if not TRANSLATOR_AVAILABLE:
                st.caption("⚠️ Translation API library not found — Urdu replies use built-in fallback text.")

        # --- Quick actions ---
        with st.container(border=True):
            st.markdown('<div class="va-section-title">⚡ Quick Actions</div>', unsafe_allow_html=True)
            qa1, qa2 = st.columns(2)
            with qa1:
                if st.button("🚚 Track Order", use_container_width=True):
                    st.session_state.pending_query = "Where is my order? I need to track it."
                    st.rerun()
                if st.button("📦 Product Info", use_container_width=True):
                    st.session_state.pending_query = "Can you give me information about this product?"
                    st.rerun()
            with qa2:
                if st.button("↩️ Return & Refund", use_container_width=True):
                    st.session_state.pending_query = "I want a refund for my order."
                    st.rerun()
                if st.button("🎧 Talk to Agent", use_container_width=True):
                    st.session_state.pending_query = "I would like to talk to a human agent please."
                    st.rerun()

        # --- Today's stats (real runtime + counts, no hardcoded numbers) ---
        with st.container(border=True):
            st.markdown('<div class="va-section-title">📈 Today\'s Stats</div>', unsafe_allow_html=True)
            n_conversations = len(st.session_state.chat_history) // 2
            n_resolved = n_conversations
            avg_resp = (
                f"{sum(st.session_state.response_times) / len(st.session_state.response_times):.2f}s"
                if st.session_state.response_times else "—"
            )
            s1, s2 = st.columns(2)
            with s1:
                st.markdown(
                    f'<div class="va-stat-box"><div class="va-stat-label">Conversations</div>'
                    f'<div class="va-stat-value">{n_conversations}</div></div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="va-stat-box"><div class="va-stat-label">Languages</div>'
                    f'<div class="va-stat-value">{len(LANGUAGE_NAMES)}</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(
                    f'<div class="va-stat-box"><div class="va-stat-label">Resolved</div>'
                    f'<div class="va-stat-value">{n_resolved}</div></div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="va-stat-box"><div class="va-stat-label">Avg. Response</div>'
                    f'<div class="va-stat-value">{avg_resp}</div></div>', unsafe_allow_html=True)

        # --- Recent history preview ---
        with st.container(border=True):
            top_row = st.columns([3, 1])
            with top_row[0]:
                st.markdown('<div class="va-section-title">🕘 Recent History</div>', unsafe_allow_html=True)
            with top_row[1]:
                if st.button("View all", key="view_all_history"):
                    st.session_state.page = "Chat History"
                    st.rerun()
            recent_user_msgs = [m for m in st.session_state.chat_history if m["role"] == "user"][-3:]
            if not recent_user_msgs:
                st.caption("No history yet.")
            for m in reversed(recent_user_msgs):
                st.markdown(
                    f"<div style='display:flex; gap:0.4rem; align-items:flex-start; font-size:0.85rem; "
                    f"padding:0.3rem 0; border-bottom:1px solid #f0f0f5;'>"
                    f"<span class='va-dot-green' style='margin-top:6px; flex-shrink:0;'></span>"
                    f"<span>{m['text'][:40]}{'...' if len(m['text']) > 40 else ''}<br>"
                    f"<span style='color:#999; font-size:0.72rem;'>"
                    f"{language_flag(m['lang'])} {language_display_name(m['lang'])} · {m['time']}</span></span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# CHAT HISTORY PAGE
# ---------------------------------------------------------------------------
def render_chat_history_page():
    st.subheader("🕘 Chat History")
    if not st.session_state.chat_history:
        st.info("No conversations yet. Start one from the Dashboard.")
        return
    with st.container(border=True):
        for i, msg in enumerate(st.session_state.chat_history):
            render_chat_bubble(msg, i)
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.session_state.response_times = []
        st.rerun()


# ---------------------------------------------------------------------------
# FAQ PAGE
# ---------------------------------------------------------------------------
def render_faq_page():
    st.subheader("❓ Frequently Asked Questions")
    faqs = [
        ("Which languages are supported?",
         "English and Urdu — including Urdu typed with a Latin keyboard "
         "(Roman Urdu), such as \"Mera order kab deliver hoga?\""),
        ("How does the assistant know what language I'm speaking?",
         "It checks the script first (native Urdu text is unambiguous), then checks "
         "for common Roman-Urdu words, and only falls back to a statistical language "
         "detector for anything that's still unclear."),
        ("How does it decide the intent of my message?",
         "A TF-IDF + Logistic Regression model, trained on labeled examples per intent "
         "and language, classifies your message into one of four categories: refund, "
         "technical issue, account help, or general inquiry."),
        ("Can I see why it gave that answer?",
         "Yes — expand \"Why this answer?\" under any assistant reply to see the key "
         "terms that drove the decision and the full confidence breakdown."),
        ("How are Urdu replies generated?",
         "The base reply is written in English, then translated into Urdu using the "
         "Google Translate API (via the deep-translator library) so replies stay in sync "
         "with the English templates without maintaining two copies by hand. If the "
         "translation API is unreachable, a built-in Urdu fallback text is used instead — "
         "you'll never see a broken response."),
        ("Does this use an external AI/LLM API?",
         "It uses a translation API for Urdu replies (see above), but intent "
         "classification itself is a local scikit-learn model — your message never "
         "leaves the app for that step. Speech-to-text and text-to-speech use Google's "
         "public APIs and need an internet connection."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.write(a)


# ---------------------------------------------------------------------------
# SETTINGS PAGE (Evaluation Module lives here)
# ---------------------------------------------------------------------------
def render_settings_page():
    st.subheader("⚙️ Settings & Model Evaluation")

    st.markdown("#### Model Evaluation")
    compare_rows = []
    for name, (_, _, m) in models.items():
        compare_rows.append({
            "Model": "Logistic Regression" if name == "logreg" else "Naive Bayes",
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
        })
    compare_df = pd.DataFrame(compare_rows)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(compare_df.set_index("Model").style.format("{:.1%}"), use_container_width=True)
    with c2:
        fig = px.bar(
            compare_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Model", barmode="group",
            title="Model Comparison (Logistic Regression vs. Naive Bayes)",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Trained on {metrics['n_train']} samples across "
        f"{len(LANGUAGE_NAMES)} languages, tested on {metrics['n_test']}."
    )

    st.markdown("#### Translation API Status")
    if TRANSLATOR_AVAILABLE:
        st.success("✅ deep-translator (Google Translate API) is installed and will be used for Urdu replies when internet access is available.")
    else:
        st.warning("⚠️ deep-translator is not installed — Urdu replies will always use the built-in fallback text.")

    st.markdown("#### Preferences")
    st.selectbox("Default language", ["Auto Detect"] + list(LANGUAGE_NAMES.values()))
    st.toggle("Dark mode (visual placeholder)")
    if st.button("Clear all chat history", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.response_times = []
        st.success("History cleared.")

    with st.expander("📁 View training dataset sample"):
        st.dataframe(dataset.sample(min(20, len(dataset))))


# ---------------------------------------------------------------------------
# PROFILE PAGE
# ---------------------------------------------------------------------------
def render_profile_page():
    st.subheader("👤 Profile")
    with st.container(border=True):
        st.write("**Name:** Demo User")
        st.write("**Role:** Customer Support Analyst")
        st.write(f"**Languages configured:** {', '.join(f'{language_flag(c)} {n}' for c, n in LANGUAGE_NAMES.items())}")
        st.write(f"**Total conversations run:** {len(st.session_state.chat_history) // 2}")


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown('<p class="va-nav-title">🤖 Voice Assistant</p>', unsafe_allow_html=True)
        st.markdown('<p class="va-nav-sub">English &amp; Urdu Customer Support</p>', unsafe_allow_html=True)

        pages = ["Dashboard", "Voice Assistant", "Chat History", "FAQ", "Settings", "Profile"]
        icons = {"Dashboard": "🏠", "Voice Assistant": "🎙️", "Chat History": "💬",
                  "FAQ": "❓", "Settings": "⚙️", "Profile": "👤"}
        choice = st.radio(
            "Navigate",
            pages,
            index=pages.index(st.session_state.page) if st.session_state.page in pages else 0,
            format_func=lambda p: f"{icons[p]}  {p}",
            label_visibility="collapsed",
        )
        st.session_state.page = choice

        st.divider()
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center;">
                    <div style="font-size:2rem;">🎧</div>
                    <b>Need human support?</b>
                    <p style="color:#7a7d8c; font-size:0.82rem;">Our support team is ready to help you 24/7.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Contact Support", use_container_width=True, key="contact_support_btn"):
                st.toast("A support agent will reach out to you shortly. 🎧", icon="✅")


# ---------------------------------------------------------------------------
# MAIN -- opens directly to the Dashboard, no login step
# ---------------------------------------------------------------------------
render_sidebar()
render_topbar()

page = st.session_state.page
if page in ("Dashboard", "Voice Assistant"):
    render_dashboard()
elif page == "Chat History":
    render_chat_history_page()
elif page == "FAQ":
    render_faq_page()
elif page == "Settings":
    render_settings_page()
elif page == "Profile":
    render_profile_page()

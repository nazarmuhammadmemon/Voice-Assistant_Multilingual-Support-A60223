# Voice Assistant — English & Urdu Customer Support

A Streamlit app styled as a product-grade voice assistant:
sidebar navigation, a chat interface with a microphone recorder, live
language detection (including Roman Urdu), an intent classifier with
plain-language explanations, quick-action shortcuts, and usage stats.

## Supported Languages
**English** and **Urdu** — including Urdu typed with a Latin keyboard
("Roman Urdu"), e.g. `"Mera order kab deliver hoga?"`, not just native
Urdu script (اردو). The interface auto-detects the language of every
message; no manual switch is required.

## Features
- **No login step** — the app opens directly to the Dashboard.
- **Sidebar navigation** — Dashboard, Voice Assistant, Chat History, FAQ, Settings, Profile.
- **Voice input** — record directly from the browser microphone (`st.audio_input`) or type a message.
- **Problem Setup Module** — validates empty/unintelligible audio and empty text before running the pipeline.
- **Core Logic Module** — modular pipeline: transcription → language detection → intent classification → response generation (English-authored, translated to Urdu), kept in `logic.py`, separate from the UI.
- **Visual UI Module** — chat bubbles with flag tags, a language card, quick-action buttons, live stats, and a Plotly confidence chart per response. Every card uses Streamlit's native `st.container(border=True)` rather than hand-written HTML `<div>` wrappers (see "UI implementation note" below).
- **Explainability Module** — every assistant reply has an expandable "Why this answer?" panel showing the key terms that drove the classification.
- **Evaluation Module** (Settings page) — accuracy / precision / recall shown live, comparing two models (Logistic Regression vs. Naive Bayes) plus real measured response latency.

## UI implementation note


1. **Custom HTML chat bubbles produced literal `<div>` text.** The chat
   bubbles were originally built as one big multi-line HTML string per
   message. When an optional piece of that string (e.g. the "translated"
   note, which is empty for most messages) substituted to an empty
   string, it left a whitespace-only line in the middle of the HTML
   block. Markdown treats a blank line as a block separator, so
   everything after it got parsed as a *new* block — and since it was
   still indented, that remainder rendered as a literal code block
   instead of live HTML. **Fix:** chat messages now use Streamlit's
   native `st.chat_message` + `st.write`/`st.caption` instead of
   hand-written HTML, which removes this entire failure mode.
2. **`StreamlitDuplicateElementId` on the confidence chart.** Every
   assistant reply renders a Plotly bar chart; without an explicit
   `key`, multiple charts in the same run collide on the same
   auto-generated ID. **Fix:** each chart now gets a unique
   `key=f"chart_{idx}"` based on the message's position in the history.

Beyond chat messages, every "card" in the UI (mic panel, language card,
quick actions, stats, recent history, etc.) uses Streamlit's native
`st.container(border=True)` rather than hand-written `<div>` wrappers —
opening a `<div>` in one `st.markdown()` call and closing it in another
doesn't actually nest anything, since each `st.markdown()` call is its
own independent element on the page. Any HTML that remains is limited to
small, fully self-contained, single-purpose snippets with no optional
pieces that could go blank.

## Project Structure
```
VoiceAssistant_Project/
├── app.py                 # Streamlit UI (entry point)
├── styles.py                # custom CSS for the product-style theme
├── logic.py                  # core pipeline: audio, language, response, explanation
├── model.py                    # data loading, preprocessing, training, evaluation
├── generate_dataset.py         # script used to build the sample dataset
├── data/
│   └── intents.csv              # sample bilingual intent-labeled dataset (176 rows: English + Urdu native script + Roman Urdu)
├── requirements.txt
├── README.md
└── screenshots/                  # add your screenshots here before submission
```

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   `st.audio_input` (the in-browser microphone recorder) requires
   **Streamlit 1.35 or newer** — `requirements.txt` already pins this.

## Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## How to Use
1. Open the app — you land directly on the **Dashboard**, no login required.
2. Click the microphone and speak, or type a message in English or Urdu (native script or Roman) in the chat box at the bottom.
3. The assistant detects your language, classifies your intent, and replies in text + speech — Urdu replies are translated live from the English template (see "Translation API" below) and marked **(Translated from English)**.
4. Try the **Quick Actions** (Track Order, Return & Refund, Product Info, Talk to Agent) in the right-hand panel for instant sample queries.
5. Visit **Chat History** for the full transcript, **FAQ** for common questions, **Settings** for model evaluation and translation-API status, and **Profile** for a demo user summary.

## AI & API Components Used
- **Intent classification (Machine Learning):** TF-IDF (whitespace tokenizer, so Urdu script is handled correctly) + Logistic Regression (primary) and Multinomial Naive Bayes (comparison baseline), trained locally with scikit-learn on `data/intents.csv`. This runs entirely on-device — no API call, no data leaves the app for this step.
- **Language detection:** rule-based, in this order:
  1. Script check — Arabic-range Unicode characters mean native Urdu.
  2. Roman-Urdu keyword check — a small list of common Urdu words spelled in Latin script (e.g. `mera`, `chahiye`, `hoga`), since Roman Urdu has no distinctive script for a statistical model to key off.
  3. `langdetect` as a final fallback.
- **Translation API:** Urdu replies are generated by translating the English response template using the **Google Translate API** (via the `deep-translator` library). This is the "use an API" component of the pipeline. **What is sent:** only the short, canned English reply text (never the user's original message). **What comes back:** the Urdu translation of that reply. **Limitations/cost/privacy:** the public Google Translate endpoint used by `deep-translator` is free and unofficial (no API key), can be rate-limited, and requires internet access; if the call fails for any reason, the app falls back to a hand-written Urdu translation of the same reply so the user never sees a broken response (see `RESPONSES_UR_FALLBACK` in `logic.py`).
- **Speech-to-text:** Google Web Speech API via `SpeechRecognition`, tried in English first, then Urdu, since a single recording could be in either language.
- **Text-to-speech:** `gTTS`, mapped per detected language (`en`, `ur`).


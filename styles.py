"""
styles.py
Custom CSS injected into the Streamlit app to match the target UI design:
a white/light theme with an indigo-violet accent, rounded cards, soft
shadows, and chat-bubble styling.
"""

CUSTOM_CSS = """
<style>
:root {
    --accent: #6C5CE7;
    --accent-dark: #5546c9;
    --accent-light: #f1eefe;
    --card-bg: #ffffff;
    --page-bg: #f6f5fb;
    --text-main: #1f2233;
    --text-muted: #7a7d8c;
    --border-soft: #ececf5;
}

.stApp {
    background-color: var(--page-bg);
}

/* Hide default Streamlit chrome for a cleaner "product" feel */
#MainMenu, footer, header {visibility: hidden;}

/* ---------- Top bar ---------- */
.va-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.2rem 1rem 0.2rem;
}
.va-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.va-brand-icon {
    font-size: 1.8rem;
    background: var(--accent-light);
    border-radius: 12px;
    padding: 0.35rem 0.55rem;
}
.va-brand-title {
    font-weight: 700;
    font-size: 1.25rem;
    color: var(--text-main);
    margin: 0;
}
.va-brand-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 0;
}

/* ---------- Sidebar nav ---------- */
section[data-testid="stSidebar"] {
    background: var(--card-bg);
    border-right: 1px solid var(--border-soft);
}
.va-nav-title {
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--text-main);
    margin-bottom: 0.2rem;
}
.va-nav-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
}

/* ---------- Native Streamlit bordered containers styled as cards ----------
   Every "card" in the UI now uses st.container(border=True) instead of a
   hand-written <div>, so this single rule gives all of them the rounded,
   shadowed look consistently and safely (no unmatched/floating tags). */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    border: 1px solid var(--border-soft) !important;
    box-shadow: 0 2px 10px rgba(108, 92, 231, 0.06);
    background: var(--card-bg);
}

/* ---------- Small reusable status dot ---------- */
.va-dot-green {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
}

/* ---------- Mic circle / listening status (used inside a bordered container) ---------- */
.va-mic-circle {
    width: 74px;
    height: 74px;
    border-radius: 50%;
    background: var(--accent-light);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 0.7rem auto;
    font-size: 2rem;
}
.va-mic-caption {
    font-weight: 600;
    color: var(--text-main);
    margin-bottom: 0.15rem;
}
.va-mic-status {
    color: var(--accent);
    font-size: 0.85rem;
}

/* ---------- Native chat message styling ---------- */
[data-testid="stChatMessage"] {
    background: var(--card-bg);
    border-radius: 14px;
    padding: 0.4rem 0.2rem;
}

/* ---------- Top bar avatar + online status dot ---------- */
.va-avatar-wrap {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 40px;
}
.va-topbar-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: var(--accent-light);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}
.va-online-dot {
    position: absolute;
    bottom: -2px;
    right: -2px;
    width: 11px;
    height: 11px;
    background: #22c55e;
    border-radius: 50%;
    border: 2px solid white;
}

/* ---------- Default (secondary) buttons: rounder, light card look ---------- */
.stButton button {
    border-radius: 12px;
    border: 1px solid var(--border-soft);
    background: #fafaff;
    color: var(--text-main);
}
.stButton button:hover {
    border-color: var(--accent);
    color: var(--accent-dark);
}

/* ---------- Stats ---------- */
.va-stat-box {
    background: var(--card-bg);
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.6rem;
}
.va-stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
}
.va-stat-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-main);
}

/* ---------- Section headers ---------- */
.va-section-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.6rem;
}

/* Buttons everywhere: rounder + accent color for primary */
.stButton>button[kind="primary"] {
    background-color: var(--accent);
    border-color: var(--accent);
    border-radius: 10px;
}
.stButton>button[kind="primary"]:hover {
    background-color: var(--accent-dark);
    border-color: var(--accent-dark);
}
</style>
"""

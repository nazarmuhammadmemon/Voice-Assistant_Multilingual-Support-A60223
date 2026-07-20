"""
logic.py
Core AI logic pipeline for the Voice Assistant (English + Urdu).
Kept separate from the Streamlit UI (app.py) as required by the lab guide.
"""

import re
import numpy as np
from langdetect import detect, DetectorFactory
from gtts import gTTS

DetectorFactory.seed = 42  # deterministic language detection

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------
LANGUAGE_NAMES = {
    "en": "English",
    "ur": "Urdu",
}
LANGUAGE_FLAGS = {
    "en": "🇬🇧",
    "ur": "🇵🇰",
}


def language_display_name(code):
    """Return a friendly display name for a language code, falling back to the code itself."""
    return LANGUAGE_NAMES.get(code, code.upper() if code else "Unknown")


def language_flag(code):
    return LANGUAGE_FLAGS.get(code, "🏳️")


# A short list of very common Roman-Urdu words/particles. This is a
# rule-based check (Option 1 in the lab guide: facts + simple inference)
# that runs before the statistical detector, because langdetect has no
# reliable way to recognise Urdu written with a Latin keyboard -- it has
# no distinctive script to key off, so it tends to guess Indonesian,
# Malay, or English instead. If enough of these tokens appear in a
# message, we call it Roman Urdu.
ROMAN_URDU_KEYWORDS = {
    "mera", "meri", "mujhe", "mujh", "aap", "ap", "kya", "kaise", "kab",
    "kahan", "hai", "hain", "raha", "rahi", "rahe", "chahiye", "chahta",
    "chahti", "nahi", "nahin", "karoon", "karna", "kar", "dain", "dijiye",
    "wapas", "paisa", "paise", "hoga", "hogi", "jayega",
    "jayegi", "bhool", "gaya", "gayi", "tabdeel", "pohncha", "poh",
    "auqaat", "dukaan", "maloomat", "shukriya",
}


def transcribe_audio(audio_path, language="en-US"):
    """
    Convert a .wav audio file to text using Google's speech recognition
    (via the SpeechRecognition library). Returns None if unintelligible.

    Tries the requested language first, then falls back to the other
    supported language, since a single recording could be in either
    English or Urdu and Google's API needs a language hint.
    """
    if not SPEECH_AVAILABLE:
        raise RuntimeError("speech_recognition library is not installed.")

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    fallback_lang = "ur-PK" if language == "en-US" else "en-US"
    for lang_code in (language, fallback_lang):
        try:
            return recognizer.recognize_google(audio_data, language=lang_code)
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            raise RuntimeError(f"Speech recognition service error: {e}")
    return None


def detect_language(text):
    """
    Detect whether a message is English or Urdu.

    Order of checks:
    1. Script check -- if the text contains Arabic-script characters,
       it's Urdu written natively (اردو), which is unambiguous.
    2. Roman-Urdu keyword check -- a simple rule-based pass over common
       Urdu words spelled with a Latin keyboard (e.g. "mera order kab
       deliver hoga"), since this is how many people actually type Urdu
       in chat apps and langdetect can't key off a distinct script here.
    3. Statistical fallback -- langdetect, used only to catch cases the
       two rules above missed. Anything that isn't Urdu defaults to
       English, since English and Urdu are the only two languages this
       assistant supports.
    """
    if not text or not text.strip():
        return "unknown"

    if any("\u0600" <= ch <= "\u06ff" for ch in text):
        return "ur"

    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    if words & ROMAN_URDU_KEYWORDS:
        return "ur"

    try:
        detected = detect(text)
    except Exception:
        detected = "en"

    return "ur" if detected == "ur" else "en"


def run_model_or_algorithm(text, clf, vectorizer):
    """
    Run the trained intent classifier on the input text.
    Returns (predicted_intent, confidence, {intent: probability, ...})
    """
    vec = vectorizer.transform([text])
    probs = clf.predict_proba(vec)[0]
    classes = clf.classes_
    top_idx = int(np.argmax(probs))
    intent = classes[top_idx]
    confidence = float(probs[top_idx])
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    return intent, confidence, prob_dict


def generate_explanation(text, intent, vectorizer, clf, top_n=3):
    """
    Produce a short natural-language explanation of why the model chose
    this intent, based on the highest-weighted TF-IDF terms present in
    the input text (Explainability Module requirement).
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    vec = vectorizer.transform([text]).toarray()[0]

    nonzero_idx = np.where(vec > 0)[0]
    if len(nonzero_idx) == 0:
        return f"Classified as '{intent}', but no strong keywords were found in the input."

    sorted_idx = nonzero_idx[np.argsort(-vec[nonzero_idx])][:top_n]
    top_terms = feature_names[sorted_idx]

    terms_str = ", ".join(f"'{t}'" for t in top_terms)
    return f"Classified as '{intent}' mainly because of these key terms: {terms_str}."


# ---------------------------------------------------------------------------
# English response templates (the single source of truth). Urdu replies
# are produced by translating these -- via a real translation API when
# available, falling back to a hand-written translation if the API call
# fails (e.g. no internet access).
# ---------------------------------------------------------------------------
RESPONSES_EN = {
    "refund": "I understand you'd like a refund. I've forwarded your request to our billing team.",
    "technical_issue": "Sorry you're running into a technical issue. Let's start troubleshooting together.",
    "account_help": "I can help you manage your account settings. Let me guide you through it.",
    "general_inquiry": "Thanks for reaching out. Here is the information you asked about.",
    "fallback": "Thank you for your message. Let me connect you to an agent.",
}

# Hand-written fallback translations, used only if the live translation
# API call fails (e.g. this sandbox has no outbound access to
# translate.google.com; your own machine most likely will).
RESPONSES_UR_FALLBACK = {
    "refund": "میں سمجھتا ہوں کہ آپ ریفنڈ چاہتے ہیں۔ میں نے آپ کی درخواست بلنگ ٹیم کو بھیج دی ہے۔",
    "technical_issue": "تکنیکی مسئلے کے لیے معذرت۔ آئیے مل کر اسے حل کرتے ہیں۔",
    "account_help": "میں آپ کے اکاؤنٹ کی سیٹنگز میں مدد کر سکتا ہوں۔",
    "general_inquiry": "رابطہ کرنے کا شکریہ۔ یہ رہی آپ کی مطلوبہ معلومات۔",
    "fallback": "آپ کے پیغام کا شکریہ۔ میں آپ کو ایک ایجنٹ سے جوڑ رہا ہوں۔",
}

TTS_LANG_CODES = {"en": "en", "ur": "ur"}


def generate_response(intent, language="en"):
    """
    Return a reply appropriate to the detected intent/language.
    Returns (response_text, was_translated_via_api: bool)
    """
    english_text = RESPONSES_EN.get(intent, RESPONSES_EN["fallback"])

    if language != "ur":
        return english_text, False

    if TRANSLATOR_AVAILABLE:
        try:
            translated = GoogleTranslator(source="en", target="ur").translate(english_text)
            if translated:
                return translated, True
        except Exception:
            pass  # fall through to the hand-written fallback below

    return RESPONSES_UR_FALLBACK.get(intent, RESPONSES_UR_FALLBACK["fallback"]), False


def synthesize_speech(text, lang="en", out_path="response_audio.mp3"):
    """Convert text to speech and save as an mp3 file. Returns the file path."""
    lang_code = TTS_LANG_CODES.get(lang, "en")
    tts = gTTS(text=text, lang=lang_code)
    tts.save(out_path)
    return out_path

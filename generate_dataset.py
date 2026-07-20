"""
generate_dataset.py
Builds the sample bilingual (English + Urdu) customer-support intent
dataset used to train the clssifier. Urdu is included in BOTH native
script (اردو) and common Roman-Urdu transliteration (e.g. "mera order
kab deliver hoga"), since real users often type Urdu using a Latin
keyboard in chat apps.

Run:
    python3 generate_dataset.py
"""

import pandas as pd
import random

random.seed(42)

templates = {
    "refund": {
        "en": [
            "I want a refund for my order",
            "Can I get my money back for this purchase",
            "Please refund my recent payment",
            "I need a refund, the product was defective",
            "How do I request a refund",
            "I was charged twice, please refund one payment",
            "The item never arrived, I want a refund",
            "Refund my subscription immediately",
            "I am not satisfied and want my money back",
            "Please process a refund for order 4521",
            "I cancelled my order, when will I get refunded",
            "The refund has not appeared in my account yet",
        ],
        "ur_native": [
            "میں اپنے آرڈر کا ریفنڈ چاہتا ہوں",
            "براہ کرم میری ادائیگی واپس کریں",
            "پراڈکٹ خراب تھا، مجھے ریفنڈ چاہیے",
            "ریفنڈ کیسے مانگوں",
            "مجھ سے دو بار پیسے کٹ گئے، ایک واپس کریں",
            "سامان نہیں پہنچا، ریفنڈ چاہیے",
            "میری سبسکرپشن کا ریفنڈ کریں",
            "میں مطمئن نہیں ہوں، پیسے واپس چاہییں",
        ],
        "ur_roman": [
            "mera order ka refund chahiye",
            "paise wapas kar dain please",
            "product kharab tha, refund chahiye",
            "refund kaise mangoon",
            "mujh se do baar paise kat gaye, ek refund kar dain",
            "saman nahi pohncha, refund chahiye",
            "meri subscription ka refund kar dain",
            "mein satisfied nahi hoon, paise wapas chahiye",
        ],
    },
    "technical_issue": {
        "en": [
            "My internet is not working",
            "The app keeps crashing on my phone",
            "I cannot log into my account",
            "The website is showing an error message",
            "My device will not connect to wifi",
            "The software update failed to install",
            "I am getting a server error when I try to pay",
            "The screen freezes every time I open the app",
            "My password reset link is not working",
            "The page keeps loading and never finishes",
            "I lost connection during the video call",
            "The app is very slow today",
        ],
        "ur_native": [
            "میرا انٹرنیٹ کام نہیں کر رہا",
            "ایپ بار بار بند ہو جاتی ہے",
            "میں اپنے اکاؤنٹ میں لاگ ان نہیں کر پا رہا",
            "ویب سائٹ پر ایرر آ رہا ہے",
            "میرا ڈیوائس وائی فائی سے کنیکٹ نہیں ہو رہا",
            "اپڈیٹ انسٹال نہیں ہو رہی",
            "ادائیگی کرتے وقت سرور ایرر آ رہا ہے",
            "اسکرین بار بار فریز ہو جاتی ہے",
        ],
        "ur_roman": [
            "mera internet kaam nahi kar raha",
            "app baar baar band ho jati hai",
            "mein apne account mein login nahi kar pa raha",
            "website par error aa raha hai",
            "mera device wifi se connect nahi ho raha",
            "update install nahi ho raha",
            "payment karte waqt server error aa raha hai",
            "screen baar baar freeze ho jati hai",
        ],
    },
    "account_help": {
        "en": [
            "How do I change my password",
            "I want to update my email address",
            "How can I delete my account",
            "I need to change my shipping address",
            "How do I update my payment method",
            "Can you help me change my username",
            "I forgot my account password",
            "How do I enable two factor authentication",
            "I want to update my phone number on file",
            "How do I close my account permanently",
            "Can I merge two accounts into one",
            "I need help verifying my identity",
        ],
        "ur_native": [
            "میں اپنا پاسورڈ کیسے تبدیل کروں",
            "میں اپنا ای میل ایڈریس اپڈیٹ کرنا چاہتا ہوں",
            "میں اپنا اکاؤنٹ کیسے بند کروں",
            "مجھے اپنا پتہ تبدیل کرنا ہے",
            "ادائیگی کا طریقہ کیسے بدلوں",
            "میں اپنا یوزرنیم تبدیل کرنا چاہتا ہوں",
            "میں اپنا پاسورڈ بھول گیا",
            "ٹو فیکٹر آتھینٹیکیشن کیسے آن کروں",
        ],
        "ur_roman": [
            "mein apna password kaise tabdeel karoon",
            "mein apna email address update karna chahta hoon",
            "mein apna account kaise band karoon",
            "mujhe apna address tabdeel karna hai",
            "payment ka tareeqa kaise badloon",
            "mein apna username tabdeel karna chahta hoon",
            "mein apna password bhool gaya",
            "two factor authentication kaise on karoon",
        ],
    },
    "general_inquiry": {
        "en": [
            "What are your business hours",
            "Do you ship internationally",
            "What payment methods do you accept",
            "How long does delivery usually take",
            "Where is your nearest store located",
            "Do you offer student discounts",
            "What is your return policy",
            "Can I track my order status",
            "Do you have a loyalty program",
            "What languages does support offer",
            "Is there a mobile app available",
            "How do I contact customer service",
            "Where is my order, can you track it for me",
            "I need information about this product",
            "Can you give me information about this product",
            "I want to talk to a human agent",
            "Can I speak to a customer service agent",
            "I would like to talk to a human agent please",
        ],
        "ur_native": [
            "آپ کے کام کے اوقات کیا ہیں",
            "کیا آپ بین الاقوامی شپنگ کرتے ہیں",
            "آپ کون سے ادائیگی کے طریقے قبول کرتے ہیں",
            "ڈیلیوری میں کتنا وقت لگتا ہے",
            "آپ کی قریبی دکان کہاں ہے",
            "کیا آپ طلباء کو رعایت دیتے ہیں",
            "آپ کی ریٹرن پالیسی کیا ہے",
            "میں اپنا آرڈر کیسے ٹریک کروں",
        ],
        "ur_roman": [
            "aap ke kaam ke auqaat kya hain",
            "kya aap international shipping karte hain",
            "aap konse payment ke tareeqe qabool karte hain",
            "delivery mein kitna waqt lagta hai",
            "aap ki qareebi dukaan kahan hai",
            "kya aap students ko discount dete hain",
            "aap ki return policy kya hai",
            "mein apna order kaise track karoon",
            "mera order kab deliver hoga",
            "mera refund kab milega",
            "product warranty kya hai",
            "mujhe product ke baray mein maloomat chahiye",
        ],
    },
}

rows = []
for intent, langs in templates.items():
    for lang_key, phrases in langs.items():
        # Both ur_native and ur_roman are stored under the single
        # language label "ur" -- the detector distinguishes the *script*
        # at runtime, but for classification purposes they're the same
        # target language.
        lang_label = "en" if lang_key == "en" else "ur"
        for p in phrases:
            rows.append({"text": p, "language": lang_label, "intent": intent})

# Slight variation for English only, to enlarge the dataset a bit further.
extra = []
suffixes_en = ["please", "thank you", "as soon as possible", "today"]
for r in rows:
    if r["language"] == "en":
        suf = random.choice(suffixes_en)
        extra.append({"text": f"{r['text']} {suf}", "language": r["language"], "intent": r["intent"]})

rows += extra
random.shuffle(rows)

df = pd.DataFrame(rows)
df.to_csv("data/intents.csv", index=False)
print(f"Saved {len(df)} rows")
print(df["intent"].value_counts())
print(df["language"].value_counts())

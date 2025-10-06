from openai import OpenAI
import os


# Prefer env var; fallback to config.py if present
try:
    from config import OPENAI_API_KEY as CONFIG_KEY
except Exception:
    CONFIG_KEY = None

def _get_api_key() -> str:
    """
    Returns the OpenAI API key.
    """
    key = os.getenv("OPENAI_API_KEY") or CONFIG_KEY
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (env or config.py).")
    return key


# -----------------------------
# Input loading (TXT / JSON only)
# -----------------------------
def load_medical_file(file_path: str) -> str:
    """
    Reads a medical text file (.txt) and returns its content as a string.
    """
    if not file_path.lower().endswith(".txt"):
        raise ValueError("Only .txt files are supported.")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()



# -----------------------------
# Prompt builder 
# -----------------------------
def create_prompt(text: str) -> str:
    
    return f"""
קלט (טקסט מקורי):
{text}

הוראות:
אתה מסכם רשומה רפואית עבור פרמדיק בשטח. הפלט חייב להיות קצר, רשימתי ובר־סריקה.
חלק את הפלט לשלושה מקטעים בדיוק, בסדר הבא: 🟥, 🟧, 🟩.
בכל מקטע הצג ארבע תתי־קטגוריות קבועות:
- אלרגיות/רגישויות
- מחלות רקע
- תרופות קבועות
- הנחיות/מגבלות

כללים מחייבים:
- כתוב נקודות בלבד (bullet points), ללא פסקאות חופשיות.
- אל תמציא עובדות. אם מידע חסר – כתוב "אין מידע".
- שמור על סדר הכותרות והתתי־כותרות בדיוק כפי שמופיע להלן.

פלט (בדיוק במבנה הבא):

🟥 גורמי סיכון מיידיים
- אלרגיות/רגישויות:
  - אין מידע
- מחלות רקע:
  - אין מידע
- תרופות קבועות:
  - אין מידע
- הנחיות/מגבלות:
  - אין מידע

🟧 היסטוריה רפואית רלוונטית
- אלרגיות/רגישויות:
  - אין מידע
- מחלות רקע:
  - אין מידע
- תרופות קבועות:
  - אין מידע
- הנחיות/מגבלות:
  - אין מידע

🟩 מידע כללי
- אלרגיות/רגישויות:
  - אין מידע
- מחלות רקע:
  - אין מידע
- תרופות קבועות:
  - אין מידע
- הנחיות/מגבלות:
  - אין מידע
""".rstrip()


# -----------------------------
# GPT call 
# -----------------------------
def summarize_medical_file(file_path: str,
                           model_name: str,
                           simulation: bool = False) -> str:
    """
    Sends the medical file content to GPT and returns a structured summary.
    If simulation=True, returns a deterministic mock without API calls.
    """
    text = load_medical_file(file_path)
    prompt = create_prompt(text)

    if simulation:
        # Deterministic mock to test formatting without API
        return (
            "🟥 גורמי סיכון מיידיים\n"
            "- אלרגיות/רגישויות:\n  - אלרגיה לפניצילין\n"
            "- מחלות רקע:\n  - אין מידע\n"
            "- תרופות קבועות:\n  - אין מידע\n"
            "- הנחיות/מגבלות:\n  - אין להכניס עירוי ביד שמאל\n\n"
            "🟧 היסטוריה רפואית רלוונטית\n"
            "- אלרגיות/רגישויות:\n  - רגישות ליוד\n"
            "- מחלות רקע:\n  - סוכרת סוג 2\n"
            "- תרופות קבועות:\n  - מטפורמין\n"
            "- הנחיות/מגבלות:\n  - אין מידע\n\n"
            "🟩 מידע כללי\n"
            "- אלרגיות/רגישויות:\n  - אין מידע\n"
            "- מחלות רקע:\n  - כאבי גב כרוניים\n"
            "- תרופות קבועות:\n  - אין מידע\n"
            "- הנחיות/מגבלות:\n  - נעזר בקביים"
        )

    client = OpenAI(api_key=_get_api_key())

    
    response = client.chat.completions.create(
        model=model_name,               # LLM version (e.g., "gpt-4o-mini" for cheap & reliable)
        messages=[                      # Conversation turns; here we only send a single user turn
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,                # 0.0 = deterministic output; critical for strict clinical formatting
        top_p=1.0,                      # Keep 1.0 when temperature=0. it means consider all kind of outputs.
        frequency_penalty=0.0,          # Discourage repetition (0.0 keeps neutral behavior)
        presence_penalty=0.0,           # Encourage novelty (0.0 prevents drifting from the structure)
        max_tokens=1000                 # Upper bound for output length/cost control
    )

    summary = response.choices[0].message.content
    return summary


# -----------------------------
# Save output
# -----------------------------
def save_summary_to_file(summary: str, output_path: str = "summary_output.txt"):
    """
    Saves the summary to a text file (overwrites if exists).
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

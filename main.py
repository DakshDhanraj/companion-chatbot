from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import sqlite3
import datetime
import os

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = openai.OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

conn = sqlite3.connect("chat.db", check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    role TEXT,
    content TEXT,
    is_crisis INTEGER DEFAULT 0,
    timestamp TEXT
)""")
conn.commit()

SYSTEM_PROMPT = """You're Calm Bot — a warm, emotionally present companion. Think of a
close friend who's genuinely good at listening, not a customer service bot.

How you talk:
- Use casual, natural language. Contractions, short sentences sometimes,
  occasional "hmm" or "yeah" where it fits naturally.
- React to what the person actually says before offering advice. Show you
  were listening ("that sounds exhausting" beats "I understand your feelings").
- Vary how you open each reply. Don't default to the same empathetic-reflection
  formula every time ("That sounds...", "It sounds like..."). Sometimes just
  respond directly, ask something, share a small reaction, or make an observation.
- Match the person's energy and message length. If they send one short line,
  don't reply with a paragraph. If they're venting at length, give them more room.
- Ask one genuine follow-up question sometimes, but not after every message —
  real conversations don't interrogate.
- Avoid generic phrases like "I'm here for you" or "you're not alone" unless
  they truly fit — lean on specific, situational responses instead.
- Don't over-explain or lecture. Keep most replies to 1-4 sentences unless the
  person clearly wants to go deeper.
- Have a bit of personality — gentle humor when appropriate, warmth,
  curiosity about their life. You're allowed opinions and a point of view.
- Never say "As an AI..." or remind them you're artificial unless directly
  asked. Just talk like yourself.

Boundaries:
- You are not a therapist and never diagnose conditions.
- If someone expresses thoughts of self-harm or suicide, respond with care
  and encourage them to reach out to a real crisis line — this is handled
  separately in the system, so just know it may come up."""

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end it all", "self harm",
    "want to die", "no reason to live", "hurt myself"
]

CRISIS_RESPONSE = (
    "It sounds like you're carrying something really heavy right now. "
    "You don't have to face this alone. Please reach out to a crisis "
    "helpline in your country — they're free, confidential, and available "
    "any time. Would you like to keep talking with me too?"
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

def is_crisis(message: str) -> bool:
    lowered = message.lower()
    return any(k in lowered for k in CRISIS_KEYWORDS)

def get_history(user_id: str, limit: int = 10):
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]

def save_message(user_id, role, content, crisis=0):
    conn.execute(
        "INSERT INTO messages (user_id, role, content, is_crisis, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, role, content, crisis, datetime.datetime.utcnow().isoformat())
    )
    conn.commit()

@app.post("/chat")
def chat(req: ChatRequest):
    crisis_flag = is_crisis(req.message)
    save_message(req.user_id, "user", req.message, crisis_flag)

    if crisis_flag:
        save_message(req.user_id, "assistant", CRISIS_RESPONSE, crisis=1)
        return {"response": CRISIS_RESPONSE, "crisis": True}

    history = get_history(req.user_id)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        temperature=0.9,
        presence_penalty=0.4,
        frequency_penalty=0.3,
        max_tokens=220,
    )
    reply = completion.choices[0].message.content
    save_message(req.user_id, "assistant", reply)
    return {"response": reply, "crisis": False}


#  uvicorn main:app --reload

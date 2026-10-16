"""Voice in, decision, voice out: a caller's voice note is transcribed, Beveridge decides, and the reply is spoken.

    pip install bev-decisions faster-whisper kokoro soundfile
    python examples/14_voice_reply.py note.wav            # writes reply.wav

Speech to text: faster-whisper (MIT). Decisions: Beveridge-Large-C. Text to speech: Kokoro-82M (Apache-2.0), a stock voice.
The reply is built from Beveridge's typed answers with fixed sentences: no language model writes it. Any answer below 0.8
confidence sends the case to a person. Beveridge can make mistakes; keep a person in the loop for important decisions.
"""

import sys

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from kokoro import KPipeline

import bev

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "billing: invoices, charges, payment details", "refunds": "refunds: money back, duplicate charges",
                          "shipping": "shipping: delivery, tracking, addresses", "technical": "technical: login, app errors, bugs",
                          "other": "none of these"}},
    "urgent": {"type": "noul", "instructions": "Does the person explicitly ask for urgent or same-day handling?"},
}


def reply(answers: dict) -> str:
    team = answers["team"]["choice"]
    sure = answers["team"]["confidence"] >= 0.8 and max(answers["urgent"]["noul"], 1 - answers["urgent"]["noul"]) >= 0.8
    if team == "other" or not sure:
        return "Thanks for your message. I have passed it to a person, who will get back to you."
    urgent = answers["urgent"]["noul"] >= 0.5
    return f"Thanks for your message. I have passed this to our {team} team" + (" and marked it urgent." if urgent else ".")


def main(path: str) -> None:
    segments, _ = WhisperModel("small.en", device="cpu", compute_type="int8").transcribe(path)
    text = " ".join(s.text.strip() for s in segments)
    answers = bev.load().decide(text, QUESTIONS)
    out = reply(answers)
    audio = np.concatenate([np.asarray(r[2]) for r in KPipeline(lang_code="b")(out, voice="bm_george")])
    sf.write("reply.wav", audio, 24000)
    print("heard:", text)
    print("team:", answers["team"]["choice"], round(answers["team"]["confidence"], 3), "| urgent:", round(answers["urgent"]["noul"], 3))
    print("reply:", out, "-> reply.wav")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "note.wav")

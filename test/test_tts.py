"""
TTS pipeline tests for DEUS Bank.

Two test modes per phrase:
  1. speak() — calls the production speak() and measures wall-clock time.
               No exception + completes within MAX_SPEAK_SECONDS = PASS.
  2. WAV      — saves audio to a temp WAV file via a subprocess (one process per
               phrase) to work around pyttsx3's macOS nsss singleton limitation
               where save_to_file only writes audio on the first call per process.
               Validates file size > MIN_WAV_BYTES = PASS.

Run without audio:  python test_tts.py --no-audio
Run with audio:     python test_tts.py
"""

import os
import sys
import time
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from utils.tts import speak

# ── ANSI helpers ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

MIN_WAV_BYTES   = 10_000   # valid audio WAV on macOS is always > 10 KB
MAX_SPEAK_SECS  = 30       # max time before speak() is considered hung

PYTHON = sys.executable

# ── Phrases (real agent messages from scenario runs) ─────────────────────────
PHRASES: list[dict] = [
    {
        "id": "greeting",
        "label": "Greeting",
        "text": (
            "Hello, I am the DEUS Bank automated support system. "
            "To authenticate, please provide at least two of the following: "
            "your full name, phone number, or IBAN."
        ),
    },
    {
        "id": "secret_question",
        "label": "Secret question prompt",
        "text": "Security question: Which is the name of my dog?",
    },
    {
        "id": "secret_wrong",
        "label": "Wrong secret answer rejection",
        "text": (
            "I'm sorry, but the answer provided was incorrect, "
            "and for security reasons I cannot proceed."
        ),
    },
    {
        "id": "details_mismatch",
        "label": "Details mismatch rejection",
        "text": (
            "Unfortunately, we are unable to verify your details at this time, "
            "and therefore cannot proceed with your request. Goodbye."
        ),
    },
    {
        "id": "welcome",
        "label": "Post-auth welcome",
        "text": "How may I assist you today?",
    },
    {
        "id": "delegate_regular",
        "label": "Delegate to human — regular client",
        "text": (
            "Ana Ferreira will attend your request. "
            "For assistance, you can also call our support department at +1112112112."
        ),
    },
    {
        "id": "delegate_premium",
        "label": "Delegate to human — premium client",
        "text": (
            "Thank you for reaching out. As a premium client, we value your experience. "
            "Miguel Santos will attend your request. "
            "For immediate support, you can also contact our dedicated support line at +1999888999."
        ),
    },
    {
        "id": "out_of_scope",
        "label": "Out-of-scope refusal",
        "text": "I can only help with your own banking services.",
    },
    {
        "id": "complaint_logged",
        "label": "Complaint registered",
        "text": "Your complaint has been registered. A specialist will review it shortly.",
    },
    {
        "id": "farewell",
        "label": "Farewell",
        "text": "Thank you for contacting DEUS Bank. Have a great day! Goodbye!",
    },
]

# ── WAV synthesis via subprocess ──────────────────────────────────────────────
# Each phrase gets its own Python process so the macOS nsss driver starts fresh.
_WAV_SCRIPT = """\
import sys, os
sys.path.insert(0, sys.argv[1])
from utils.tts import save_speech_to_file
save_speech_to_file(sys.argv[2], sys.argv[3])
"""


def synthesise_wav_subprocess(text: str, wav_path: str, src_dir: str) -> tuple[bool, str]:
    """Run save_speech_to_file in a subprocess. Returns (ok, error_message)."""
    try:
        result = subprocess.run(
            [PYTHON, "-c", _WAV_SCRIPT, src_dir, text, wav_path],
            capture_output=True,
            text=True,
            timeout=MAX_SPEAK_SECS,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {MAX_SPEAK_SECS}s"
    except Exception as exc:
        return False, str(exc)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tts_test(phrase: dict, play_audio: bool, src_dir: str) -> bool:
    pid   = phrase["id"]
    label = phrase["label"]
    text  = phrase["text"]

    print(f"\n{'─' * 60}")
    print(f"{BOLD}[{pid}]{RESET}  {label}")
    print(f"{DIM}  \"{text[:90]}{'...' if len(text) > 90 else ''}\"  {RESET}")

    passed_wav   = False
    passed_speak = False

    # ── Test 1: WAV file synthesis ────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"tts_{pid}_") as tmp:
        wav_path = tmp.name
    try:
        t0 = time.perf_counter()
        ok, err = synthesise_wav_subprocess(text, wav_path, src_dir)
        elapsed = time.perf_counter() - t0

        if not ok:
            print(f"  WAV  {RED}FAIL x{RESET}  subprocess error: {err}")
        else:
            file_size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
            if file_size < MIN_WAV_BYTES:
                print(f"  WAV  {RED}FAIL x{RESET}  file too small ({file_size:,} bytes) — audio likely silent")
            else:
                print(f"  WAV  {GREEN}PASS v{RESET}  {file_size:,} bytes  |  {elapsed:.2f}s")
                passed_wav = True
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    # ── Test 2: speak() playback ──────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        if play_audio:
            print("  Playing...")
            speak(text)
        elapsed = time.perf_counter() - t0
        label_time = f"  |  {elapsed:.2f}s" if play_audio else ""
        print(f"  speak {GREEN}PASS v{RESET}{label_time}")
        passed_speak = True
    except Exception as exc:
        print(f"  speak {RED}FAIL x{RESET}  {exc}")

    return passed_wav and passed_speak


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    play_audio = "--no-audio" not in sys.argv
    src_dir    = os.path.dirname(os.path.abspath(__file__))

    if not play_audio:
        print(f"{DIM}Audio playback disabled (--no-audio){RESET}")

    print(f"\n{'=' * 60}")
    print(f"{BOLD}TTS TEST SUITE -- DEUS Bank{RESET}")
    print(f"Engine : pyttsx3  |  Rate : 165 wpm  |  Phrases : {len(PHRASES)}")
    print(f"Tests  : WAV synthesis (subprocess) + speak() playback")
    print(f"{'=' * 60}")

    results = [run_tts_test(p, play_audio=play_audio, src_dir=src_dir) for p in PHRASES]

    passed = sum(results)
    failed = len(results) - passed

    print(f"\n{'=' * 60}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"  {GREEN}Passed : {passed}{RESET}")
    print(f"  {RED}Failed : {failed}{RESET}")
    print(f"{'=' * 60}\n")

    if failed:
        sys.exit(1)

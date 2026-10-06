import os
import requests
from dotenv import load_dotenv


# Load Telegram credentials from .env
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN was not found in .env")

if not CHAT_ID:
    raise ValueError("TELEGRAM_CHAT_ID was not found in .env")


# Evidence video to send
VIDEO_PATH = "evidence/clips/possible_concealment_2026-10-03_02-00-15.mp4"

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Evidence video not found: {VIDEO_PATH}")


url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"

caption = (
    "⚠️ AI SECURITY ALERT\n\n"
    "Possible concealment detected.\n"
    "Review the attached evidence clip."
)


with open(VIDEO_PATH, "rb") as video_file:
    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "caption": caption
        },
        files={
            "video": video_file
        },
        timeout=60
    )


if response.ok:
    print("SUCCESS: Evidence video sent to Telegram.")
else:
    print("ERROR: Evidence video was not sent.")
    print(response.text)
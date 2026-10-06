import os
import requests
from dotenv import load_dotenv


load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def telegram_configured():
    """Return True when Telegram credentials are available."""
    return bool(BOT_TOKEN and CHAT_ID)


def send_message(message):
    """Send a text message to Telegram."""

    if not telegram_configured():
        print("TELEGRAM ERROR: Credentials are missing.")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=15
        )

        if response.ok:
            print("TELEGRAM: Alert message sent.")
            return True

        print("TELEGRAM ERROR: Message was not sent.")
        print(response.text)

    except requests.RequestException as error:
        print(f"TELEGRAM ERROR: {error}")

    return False


def send_photo(photo_path, caption=None):
    """Send an evidence image to Telegram."""

    if not telegram_configured():
        print("TELEGRAM ERROR: Credentials are missing.")
        return False

    if not photo_path or not os.path.exists(photo_path):
        print(f"TELEGRAM ERROR: Image not found: {photo_path}")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:
        with open(photo_path, "rb") as photo_file:
            response = requests.post(
                url,
                data={
                    "chat_id": CHAT_ID,
                    "caption": caption or ""
                },
                files={
                    "photo": photo_file
                },
                timeout=30
            )

        if response.ok:
            print("TELEGRAM: Evidence image sent.")
            return True

        print("TELEGRAM ERROR: Image was not sent.")
        print(response.text)

    except requests.RequestException as error:
        print(f"TELEGRAM ERROR: {error}")

    return False


def send_video(video_path, caption=None):
    """Send an evidence video to Telegram."""

    if not telegram_configured():
        print("TELEGRAM ERROR: Credentials are missing.")
        return False

    if not video_path or not os.path.exists(video_path):
        print(f"TELEGRAM ERROR: Video not found: {video_path}")
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"

    try:
        with open(video_path, "rb") as video_file:
            response = requests.post(
                url,
                data={
                    "chat_id": CHAT_ID,
                    "caption": caption or ""
                },
                files={
                    "video": video_file
                },
                timeout=60
            )

        if response.ok:
            print("TELEGRAM: Evidence video sent.")
            return True

        print("TELEGRAM ERROR: Video was not sent.")
        print(response.text)

    except requests.RequestException as error:
        print(f"TELEGRAM ERROR: {error}")

    return False
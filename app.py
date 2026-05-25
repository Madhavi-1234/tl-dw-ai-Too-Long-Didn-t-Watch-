import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

from faster_whisper import WhisperModel
import yt_dlp
import glob
from urllib.parse import urlparse, parse_qs


# -----------------------------
# ENV + GEMINI
# -----------------------------
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


# -----------------------------
# PROMPT
# -----------------------------
prompt = """
You are a helpful YouTube assistant.

Summarize the transcript clearly:
- Bullet points
- Key insights
- Max 250 words
"""


# -----------------------------
# VIDEO ID (SAFE FIX)
# -----------------------------
def extract_video_id(url):
    try:
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        else:
            return parse_qs(urlparse(url).query)["v"][0]
    except:
        return None


# -----------------------------
# TRANSCRIPT FUNCTION (ROBUST)
# -----------------------------
def get_transcript_details(youtube_video_url):

    # -------- TRY YOUTUBE API --------
    try:
        video_id = extract_video_id(youtube_video_url)

        if video_id:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([i["text"] for i in transcript_list])

    except Exception:
        st.warning("Transcript API failed → using Whisper fallback")

    # -------- WHISPER FALLBACK --------
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'audio.%(ext)s',
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_video_url])

        # safer file detection
        audio_files = glob.glob("audio.*")

        if not audio_files:
            return "❌ Audio download failed"

        audio_file = audio_files[0]

        # Whisper
        whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

        segments, info = whisper_model.transcribe(audio_file)

        return " ".join([seg.text for seg in segments])

    except Exception as e:
        return f"❌ Whisper failed: {str(e)}"


# -----------------------------
# GEMINI
# -----------------------------
def get_gemini_content(transcript_text):

    if not transcript_text:
        return "❌ No transcript found"

    full_prompt = prompt + "\n\nTranscript:\n" + transcript_text

    response = model.generate_content(full_prompt)

    return response.text


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎥 YouTube Insight Bot")

youtube_link = st.text_input("Enter YouTube URL")

if st.button("Generate Insights"):

    if not youtube_link:
        st.error("Please enter URL")
    else:

        with st.spinner("Processing video... ⏳"):
            transcript_text = get_transcript_details(youtube_link)

        if "❌" in transcript_text:
            st.error(transcript_text)
        else:
            summary = get_gemini_content(transcript_text)

            st.markdown("## 📌 Insights")
            st.write(summary)
import streamlit as st
import os
import glob
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel
import yt_dlp
from urllib.parse import urlparse, parse_qs

# -----------------------------
# ENV + GEMINI
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# LOAD WHISPER ONCE (IMPORTANT)
# -----------------------------
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# -----------------------------
# PROMPT
# -----------------------------
prompt = """
You are a helpful YouTube assistant.

Summarize the transcript:
- Bullet points
- Key insights
- Keep it under 250 words
"""

# -----------------------------
# VIDEO ID
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
# TRANSCRIPT FIRST
# -----------------------------
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except:
        return None

# -----------------------------
# WHISPER FALLBACK
# -----------------------------
def whisper_transcribe(url):

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'quiet': True,
        'noplaylist': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_files = glob.glob("audio.*")

    if not audio_files:
        return None

    audio_file = audio_files[0]

    segments, _ = whisper_model.transcribe(audio_file)

    text = " ".join([seg.text for seg in segments])

    # cleanup
    os.remove(audio_file)

    return text

# -----------------------------
# MAIN PIPELINE (SMART BRAIN)
# -----------------------------
def get_text_from_video(url):

    video_id = extract_video_id(url)

    # STEP 1: transcript
    if video_id:
        text = get_transcript(video_id)
        if text:
            return text

    st.info("Transcript not available → using Whisper...")

    # STEP 2: whisper fallback
    return whisper_transcribe(url)

# -----------------------------
# GEMINI SUMMARY
# -----------------------------
def get_summary(text):

    if not text:
        return "No transcript found"

    response = model.generate_content(prompt + "\n\nTranscript:\n" + text)
    return response.text

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎥 YouTube AI Insight Bot")

url = st.text_input("Enter YouTube URL")

if st.button("Generate Insights"):

    if not url:
        st.error("Please enter URL")
    else:

        with st.spinner("Processing video..."):

            transcript = get_text_from_video(url)

            if not transcript:
                st.error("Failed to process video")
            else:
                summary = get_summary(transcript)

                st.subheader("📌 Insights")
                st.write(summary)

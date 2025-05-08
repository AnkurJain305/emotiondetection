import streamlit as st
import joblib
import neattext.functions as nfx
from tempfile import NamedTemporaryFile

# Load emotion detection model
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Try loading whisper (for local demo only)
try:
    import whisper
    whisper_model = whisper.load_model("base")
    whisper_available = True
except:
    whisper_available = False

st.title("🎤 Emotion Detection from Text or Speech")
st.write("Choose to either enter text or upload a short voice clip (.wav/.mp3).")

option = st.radio("Select input type:", ("Text", "Speech"))

# --- Text Input ---
if option == "Text":
    user_text = st.text_area("Enter your text here:")
    if user_text:
        clean_text = nfx.remove_special_characters(nfx.remove_stopwords(user_text))
        vectorized_text = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_text)[0]
        st.subheader("🔍 Predicted Emotion:")
        st.success(prediction)

# --- Speech Input ---
else:
    if not whisper_available:
        st.warning("⚠️ Whisper is not available in this environment. Please run locally for speech input.")
    else:
        audio_file = st.file_uploader("Upload a short audio file (WAV or MP3)", type=["wav", "mp3"])

        if audio_file is not None:
            with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_file.read())
                temp_audio_path = temp_audio.name

            st.audio(temp_audio_path)

            with st.spinner("Transcribing audio..."):
                result = whisper_model.transcribe(temp_audio_path)
                transcribed_text = result["text"]
                st.markdown(f"**Transcribed Text:** {transcribed_text}")

                clean_text = nfx.remove_special_characters(nfx.remove_stopwords(transcribed_text))
                vectorized_text = vectorizer.transform([clean_text])
                prediction = model.predict(vectorized_text)[0]

                st.subheader("🔍 Predicted Emotion:")
                st.success(prediction)

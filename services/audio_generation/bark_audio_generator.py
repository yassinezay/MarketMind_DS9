from transformers import AutoProcessor, BarkModel
from langdetect import detect
from scipy.io.wavfile import write
import os
import numpy as np

# Mapping combiné (langue + sexe)
VOICE_PRESETS = {
    "fr": {
        "male": "v2/fr_speaker_3",
        "female": "v2/fr_speaker_1"
    },
    "en": {
        "male": "v2/en_speaker_0",
        "female": "v2/en_speaker_9"
    }
}

def detect_language_and_select_voice(text, voice_gender):
    try:
        lang = detect(text)
        lang_code = "fr" if lang == "fr" else "en"
    except:
        lang_code = "en"

    # Assurez-vous que `voice_gender` est "male" ou "female"
    return VOICE_PRESETS[lang_code][voice_gender]



# --- FastAPI endpoint for audio generation ---
def generate_audio(text, voice_gender="male"):
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")

    voice_preset = detect_language_and_select_voice(text, voice_gender)
    inputs = processor(text, voice_preset=voice_preset)
    audio_array = model.generate(**inputs).cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate

    return audio_array, sample_rate

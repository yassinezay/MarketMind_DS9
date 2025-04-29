# bark_audio_generator.py
from transformers import AutoProcessor, BarkModel
from langdetect import detect
import torch
import numpy as np
from scipy.io.wavfile import write

def detect_language(text):
    try:
        lang = detect(text)
        if lang == "fr":
            return "v2/fr_speaker_1"
        else:
            return "v2/en_speaker_1"
    except:
        # Fallback to English if detection fails
        return "v2/en_speaker_1"

def generate_audio(text):
    voice_preset = detect_language(text)
    
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")

    inputs = processor(text, voice_preset=voice_preset)
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate

    return audio_array, sample_rate

def save_audio_to_wav(audio_array, sample_rate, filename="generated_audio.wav"):
    write(filename, sample_rate, audio_array)
    print(f"🎧 Audio saved to {filename}")

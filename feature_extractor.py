# feature_extractor.py (Corrected: Removed runtime NLTK download)

import librosa
import librosa.effects
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import collections
import os # Keep os import if needed elsewhere, but not for NLTK_DATA here

from config import (
    TARGET_SR, PAUSE_THRESHOLD_DB, MIN_SILENCE_DURATION_S,
    FMIN, FMAX, HESITATION_MARKERS
)

# --- REMOVED THE NLTK DOWNLOAD BLOCK ---
# The 'punkt' data should be pre-downloaded by the Dockerfile build process


def extract_audio_features(y, sr):
    """ Extracts acoustic features from the audio waveform. """
    print(f"   Extracting audio features...")
    features = {}
    total_duration = librosa.get_duration(y=y, sr=sr)
    if total_duration == 0: total_duration = 1e-6

    try:
        hop_length_pauses = 512
        non_silent_intervals = librosa.effects.split(y, top_db=PAUSE_THRESHOLD_DB, hop_length=hop_length_pauses)
        speech_duration = sum(librosa.samples_to_time(interval[1] - interval[0], sr=sr) for interval in non_silent_intervals)
        features['speech_fraction'] = speech_duration / total_duration
        pauses = []; last_end_time = 0
        for start, end in non_silent_intervals:
            start_time = librosa.samples_to_time(start, sr=sr); end_time = librosa.samples_to_time(end, sr=sr)
            silence_duration = start_time - last_end_time
            if silence_duration >= MIN_SILENCE_DURATION_S: pauses.append(silence_duration)
            last_end_time = end_time
        final_silence = total_duration - last_end_time
        if final_silence >= MIN_SILENCE_DURATION_S: pauses.append(final_silence)
        num_pauses = len(pauses)
        features['pause_rate'] = num_pauses / total_duration
        features['avg_pause_duration'] = np.mean(pauses) if num_pauses > 0 else 0
    except Exception as e:
        print(f"      Warning: Error in pause analysis: {e}")
        features.update({'speech_fraction': np.nan, 'pause_rate': np.nan, 'avg_pause_duration': np.nan})

    try:
        frame_length_pitch = 2048; hop_length_pitch = 512
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=frame_length_pitch, hop_length=hop_length_pitch)
        voiced_f0 = f0[voiced_flag]
        if voiced_f0 is not None and len(voiced_f0) > 1:
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
            if len(voiced_f0) > 1:
                features['pitch_mean'] = np.mean(voiced_f0); features['pitch_stddev'] = np.std(voiced_f0)
            else: features.update({'pitch_mean': np.nan, 'pitch_stddev': np.nan})
        else: features.update({'pitch_mean': np.nan, 'pitch_stddev': np.nan})
    except Exception as e:
        print(f"      Warning: Error in pitch analysis: {e}")
        features.update({'pitch_mean': np.nan, 'pitch_stddev': np.nan})

    print(f"      Audio features extracted: { {k: f'{v:.3f}' if isinstance(v, (float, np.number)) and not np.isnan(v) else v for k, v in features.items()} }")
    return features

def extract_text_features(transcript, audio_duration_seconds):
    """ Extracts linguistic features from the transcript. """
    print(f"   Extracting text features...")
    features = {}
    if audio_duration_seconds <= 0: audio_duration_seconds = 1e-6
    if not transcript or not transcript.strip():
        print("      Transcript empty, returning default/NaN text features.")
        features.update({'word_count': 0, 'speech_rate_wps': 0, 'hesitation_rate': 0, 'lexical_diversity_ttr': np.nan, 'avg_sentence_length': 0})
        return features

    # The code now assumes 'punkt' is available because the Docker build downloaded it
    try:
        words = word_tokenize(transcript.lower())
        cleaned_words = [word for word in words if word.isalnum() or '-' in word]
        words_for_hesitation = [word for word in words if word.isalnum()]
        features['word_count'] = len(cleaned_words)
        features['speech_rate_wps'] = features['word_count'] / audio_duration_seconds
        hesitation_count = sum(1 for word in words_for_hesitation if word in HESITATION_MARKERS)
        total_words_for_hes_rate = len(words_for_hesitation)
        features['hesitation_rate'] = hesitation_count / total_words_for_hes_rate if total_words_for_hes_rate > 0 else 0
        if features['word_count'] > 0: unique_words = set(cleaned_words); features['lexical_diversity_ttr'] = len(unique_words) / features['word_count']
        else: features['lexical_diversity_ttr'] = np.nan
        sentences = sent_tokenize(transcript)
        if sentences:
            sentence_lengths = [len(w) for s in sentences if (w := [word for word in word_tokenize(s.lower()) if word.isalnum() or '-' in word])]
            features['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
        else: features['avg_sentence_length'] = 0
    except Exception as e:
        # This catch block is fine, for errors during actual tokenization/feature calculation
        print(f"      Warning: Error in text feature extraction: {e}")
        features.setdefault('word_count', 0); features.setdefault('speech_rate_wps', 0); features.setdefault('hesitation_rate', 0)
        features.setdefault('lexical_diversity_ttr', np.nan); features.setdefault('avg_sentence_length', 0)

    print(f"      Text features extracted: { {k: f'{v:.3f}' if isinstance(v, (float, np.number)) and not np.isnan(v) else v for k, v in features.items()} }")
    return features
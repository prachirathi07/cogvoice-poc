import os
import librosa

BASE_DATA_DIR = r"C:\Users\prach\Desktop\MemoTag\data" 
CONTROL_FOLDER = os.path.join(BASE_DATA_DIR, "Control")
DEMENTIA_FOLDER = os.path.join(BASE_DATA_DIR, "Dementia")

OUTPUT_DIR = BASE_DATA_DIR 
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

OUTPUT_CSV_FILENAME = "features_and_transcripts.csv"
MODELS_FILENAME = 'models.pkl'
MODELS_PATH = os.path.join(MODELS_DIR, MODELS_FILENAME)

TARGET_SR = 16000 

PAUSE_THRESHOLD_DB = 10
MIN_SILENCE_DURATION_S = 0.05 
FMIN = librosa.note_to_hz('C2') 
FMAX = librosa.note_to_hz('C7') 
HESITATION_MARKERS = ["uh", "um", "ah", "eh", "er", "like", "you know", "I mean", "basically", "actually",]

RANDOM_STATE = 42


EXPECTED_AUDIO_COLS = ['speech_fraction', 'pause_rate', 'avg_pause_duration', 'pitch_mean', 'pitch_stddev']
EXPECTED_TEXT_COLS = ['word_count', 'speech_rate_wps', 'hesitation_rate', 'lexical_diversity_ttr', 'avg_sentence_length']
FEATURE_COLUMNS = EXPECTED_AUDIO_COLS + EXPECTED_TEXT_COLS
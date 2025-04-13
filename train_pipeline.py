
import os
import glob
import librosa
import whisper
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans


import config
from feature_extractor import extract_audio_features, extract_text_features

def train_and_save_models():
    """
    Processes training data, extracts features, trains models,
    and saves the pipeline components.
    """

    print("Loading Whisper model...")
    try:
        whisper_model = whisper.load_model("base")
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Error loading Whisper model: {e}"); exit()


    results = []
    audio_files = []
    control_files = glob.glob(os.path.join(config.CONTROL_FOLDER, "*.wav"))
    dementia_files = glob.glob(os.path.join(config.DEMENTIA_FOLDER, "*.wav"))
    for f in control_files: audio_files.append({"path": f, "label": "Control"})
    for f in dementia_files: audio_files.append({"path": f, "label": "Dementia"})

    if not audio_files:
        print(f"FATAL ERROR: No .wav files found in {config.CONTROL_FOLDER} or {config.DEMENTIA_FOLDER}."); exit()
    print(f"\nFound {len(audio_files)} audio files for feature extraction.")

    for i, file_info in enumerate(audio_files):
        file_path, true_label = file_info["path"], file_info["label"]
        print(f"\nProcessing Training File {i+1}/{len(audio_files)}: {os.path.basename(file_path)} (Label: {true_label})")
        audio_features, text_features, transcript, error_msg = {}, {}, "", None
        audio_duration = 0
        try:
            y, sr = librosa.load(file_path, sr=config.TARGET_SR, mono=True); audio_duration = librosa.get_duration(y=y, sr=sr)
            if audio_duration >= 0.5:
                 if y.dtype != np.float32: y = y.astype(np.float32)
                 transcription_result = whisper_model.transcribe(y, language='en', fp16=False); transcript = transcription_result['text'].strip()
                 audio_features = extract_audio_features(y, sr)
                 text_features = extract_text_features(transcript, audio_duration)
            else: print("      Audio too short, skipping.")
        except Exception as e: print(f"   !!! ERROR processing file {file_path}: {e}"); error_msg = str(e)
        file_results = {"filename": os.path.basename(file_path), "label": true_label, "duration_s": audio_duration, "transcript": transcript, **audio_features, **text_features}
        if error_msg: file_results["error"] = error_msg
        results.append(file_results)

    results_df = pd.DataFrame(results)
    output_csv_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_CSV_FILENAME)
    try: results_df.to_csv(output_csv_path, index=False, encoding='utf-8'); print(f"\nFeatures saved to: {output_csv_path}")
    except Exception as e: print(f"\nWarning: Error saving features to CSV: {e}")


    print("\nStarting Unsupervised Analysis and Model Training")


    feature_columns = config.FEATURE_COLUMNS 
    valid_feature_columns = [col for col in feature_columns if col in results_df.columns]
    if not valid_feature_columns: print("FATAL ERROR: No valid numerical feature columns found for training."); exit()
    features_df = results_df[valid_feature_columns].copy()
    print(f"Using features for training: {valid_feature_columns}")

    print(f"\nNaN count before imputation:\n{features_df.isnull().sum().to_string()}")
    imputer = SimpleImputer(strategy='median'); features_imputed_array = imputer.fit_transform(features_df)
    if features_df.isnull().values.any(): print(f"NaNs were imputed.")
    else: print("No NaNs found; imputer is fitted.")

    scaler = StandardScaler(); features_scaled_array = scaler.fit_transform(features_imputed_array)
    features_scaled_df = pd.DataFrame(features_scaled_array, columns=valid_feature_columns, index=features_df.index)
    print("\nTraining features prepared, imputed, and scaled.")

    print("\nTraining Isolation Forest...")
    iso_forest = IsolationForest(contamination='auto', random_state=config.RANDOM_STATE); iso_forest.fit(features_scaled_df)
    print("Isolation Forest training complete.")

    print("\nTraining K-Means Clustering (K=2)...")
    kmeans = KMeans(n_clusters=2, random_state=config.RANDOM_STATE, n_init=10); kmeans.fit(features_scaled_df)
    cluster_labels = kmeans.labels_
    temp_results_df = results_df.loc[features_scaled_df.index].copy() 
    temp_results_df['kmeans_cluster'] = cluster_labels
    print("K-Means training complete.")

    print("\nDetermining Dementia-like Cluster ID...")
    dementia_cluster_id = 0 
    control_cluster_id = 1
    if 'kmeans_cluster' in temp_results_df.columns and 'label' in temp_results_df.columns:
        cluster_analysis_df = temp_results_df[['label', 'kmeans_cluster']].dropna()
        if not cluster_analysis_df.empty and 'Dementia' in cluster_analysis_df['label'].unique():
            try:
                cluster_counts = cluster_analysis_df.groupby(['kmeans_cluster', 'label']).size().unstack(fill_value=0)
                print("\nCluster composition (Training Data):"); print(cluster_counts)
                if 'Dementia' in cluster_counts.columns and not cluster_counts['Dementia'].empty:
                     dementia_cluster_id = cluster_counts['Dementia'].idxmax()
                     control_cluster_id = next(iter(set(cluster_counts.index) - {dementia_cluster_id}), 1 - dementia_cluster_id)
                     print(f"Determined Dementia-like Cluster ID: {dementia_cluster_id}")
                     print(f"Determined Control-like Cluster ID: {control_cluster_id}")
                else: print("Warning: Could not determine dementia cluster. Using default.")
            except Exception as e: print(f"Warning: Could not determine dementia cluster: {e}. Using default.")
        else: print("Warning: Cannot determine dementia cluster. Using default.")
    else: print("Warning: Cannot determine dementia cluster. Using default.")

    print("\nSaving Trained Models and Components")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    models_to_save = {
        'feature_columns': valid_feature_columns,
        'imputer': imputer, 'scaler': scaler, 'isolation_forest': iso_forest,
        'kmeans': kmeans, 'dementia_cluster_id': dementia_cluster_id
    }
    try:
        with open(config.MODELS_PATH, 'wb') as f: pickle.dump(models_to_save, f)
        print(f"Models saved successfully to: {config.MODELS_PATH}")
    except Exception as e: print(f"Error saving models: {e}")

    print("\nTraining Pipeline Complete")


if __name__ == "__main__":
    train_and_save_models()
import os
import pickle
import librosa
import numpy as np
import pandas as pd
import whisper
import traceback

import config
from feature_extractor import extract_audio_features, extract_text_features


def _predict_risk_and_cluster(audio_path, whisper_model, feature_columns, imputer, scaler, iso_forest, kmeans):
    """ Internal function: processes audio, gets raw model predictions. """
    print(f"\n--- Processing for prediction: {os.path.basename(audio_path)} ---")
    results = {"transcript": None, "extracted_features": None, "anomaly_score": np.nan,
               "anomaly_prediction": None, "kmeans_cluster": None, "error": None}
    try:
        print("   Loading audio...")
        y, sr = librosa.load(audio_path, sr=config.TARGET_SR, mono=True)
        audio_duration = librosa.get_duration(y=y, sr=sr)
        if audio_duration < 0.5: raise ValueError("Audio too short (< 0.5 seconds)")
        print(f"   Audio loaded. Duration: {audio_duration:.2f} seconds.")

        print("   Transcribing with Whisper...");
        if y.dtype != np.float32: y = y.astype(np.float32)
        transcription_result = whisper_model.transcribe(y, language='en', fp16=False)
        transcript = transcription_result['text'].strip()
        results["transcript"] = transcript; print(f"   Transcript: \"{transcript[:100]}...\"")

        audio_features = extract_audio_features(y, sr)
        text_features = extract_text_features(transcript, audio_duration)
        all_features = {**audio_features, **text_features}; results["extracted_features"] = all_features

        try:
            feature_data = {col: all_features.get(col, np.nan) for col in feature_columns}
            new_features_df = pd.DataFrame([feature_data], columns=feature_columns)
        except Exception as e: raise ValueError(f"Error preparing feature vector: {e}")

        if new_features_df.isnull().values.any():
            print(f"   WARNING: NaNs detected, imputing:\n{new_features_df.isnull().sum().to_string()}")

        print("   Applying imputer..."); features_imputed_array = imputer.transform(new_features_df.values)
        print("   Applying scaler..."); features_scaled_array = scaler.transform(features_imputed_array)
        print("   Making predictions...")
        results["anomaly_score"] = float(iso_forest.decision_function(features_scaled_array)[0])
        results["anomaly_prediction"] = int(iso_forest.predict(features_scaled_array)[0])
        results["kmeans_cluster"] = int(kmeans.predict(features_scaled_array)[0])
        # Get distances to cluster centers for more nuanced scoring
        results["kmeans_distances"] = kmeans.transform(features_scaled_array)[0].tolist()
        print("   Prediction complete.")

    except Exception as e:
        results["error"] = f"An error occurred during prediction processing: {str(e)}"; print(f"   Error: {results['error']}")
    return results


def get_prediction(audio_path, models_path=config.MODELS_PATH, whisper_model_instance=None):
    """
    Loads models, processes audio, gets predictions, and calculates risk score.
    """
    print(f"\nStarting prediction pipeline for: {os.path.basename(audio_path)}")
    final_results = {"error": None, "risk_score": None, "risk_category": "Error"}

    try:
        if not os.path.exists(models_path): raise FileNotFoundError(f"Models file not found at {models_path}")
        print(f"Loading models from {models_path}...")
        with open(models_path, 'rb') as f: models = pickle.load(f)
        print("Models loaded successfully.")
        feature_columns = models['feature_columns']; imputer = models['imputer']
        scaler = models['scaler']; iso_forest = models['isolation_forest']
        kmeans = models['kmeans']; dementia_cluster_id = models.get('dementia_cluster_id', 0)
        print(f"Using Dementia-like Cluster ID: {dementia_cluster_id} (from saved model)")
    except Exception as e:
        final_results["error"] = f"Failed to load models: {e}"; print(final_results["error"]); return final_results

    whisper_to_use = whisper_model_instance
    try:
        if whisper_to_use is None:
            print("Loading Whisper model (base) for prediction...")
            whisper_to_use = whisper.load_model("base")
            print("Whisper model loaded successfully for prediction.")
    except Exception as e:
        final_results["error"] = f"Failed to load Whisper model: {e}"; print(final_results["error"]); return final_results

    raw_results = _predict_risk_and_cluster(
        audio_path=audio_path, whisper_model=whisper_to_use, feature_columns=feature_columns,
        imputer=imputer, scaler=scaler, iso_forest=iso_forest, kmeans=kmeans
    )
    final_results.update(raw_results)

    if final_results["error"] is None:
        print("\n--- Risk Assessment (Enhanced) ---")
        
        
        anomaly_score = final_results.get('anomaly_score', 0)
        normalized_anomaly_score = 1 - ((anomaly_score + 0.5) / 1.0)
        normalized_anomaly_score = max(0, min(1, normalized_anomaly_score))
        anomaly_component = normalized_anomaly_score * 50
        
        distances = final_results.get('kmeans_distances', [])
        if len(distances) >= 2:
            dementia_distance = distances[dementia_cluster_id]
            control_distance = distances[1 - dementia_cluster_id]
            total_distance = dementia_distance + control_distance
            
            if total_distance > 0:
                # Calculate proximity (inverse of distance) to the dementia cluster
                # The closer to dementia cluster (and further from control), the higher the score
                dementia_proximity = 1 - (dementia_distance / total_distance)
                # Scale to 0-50 points contribution
                cluster_component = dementia_proximity * 50
            else:
                # Fallback if distances are invalid
                cluster_component = 50 if final_results.get('kmeans_cluster') == dementia_cluster_id else 0
        else:
            # Fallback to binary approach if distances not available
            cluster_component = 50 if final_results.get('kmeans_cluster') == dementia_cluster_id else 0
        
        # ENHANCEMENT 3: Add feature-specific adjustments
        features = final_results.get('extracted_features', {})
        feature_adjustment = 0
        if features:
            feature_weights = {
                'hesitation_rate': 3.0,        # High hesitation is concerning
                'lexical_diversity_ttr': -2.0, # Lower diversity is concerning
                'pause_rate': 2.0,             # More pauses may indicate issues
                'speech_rate_wps': -1.5,       # Slower speech rate may indicate issues
                'avg_pause_duration': 1.0      # Longer pauses may indicate issues
            }
            
            for feature, weight in feature_weights.items():
                if feature in features and not np.isnan(features[feature]):
                    if feature == 'hesitation_rate':
                        norm_value = min(1.0, features[feature] * 5)  
                    elif feature == 'lexical_diversity_ttr':
                        norm_value = 1.0 - features[feature]  
                    elif feature == 'pause_rate':
                        norm_value = min(1.0, features[feature] * 2)  
                    elif feature == 'speech_rate_wps':
                        norm_value = max(0, 1.0 - (features[feature] / 3.0))  
                    elif feature == 'avg_pause_duration':
                        norm_value = min(1.0, features[feature])  
                    else:
                        norm_value = 0.5  
                    
                    feature_adjustment += norm_value * weight
            
            feature_adjustment = max(-10, min(10, feature_adjustment))
        
        risk_score = anomaly_component + cluster_component + feature_adjustment
        
        risk_score = max(0, min(100, risk_score))
        final_results['risk_score'] = risk_score
        
        print(f"Anomaly Component: {anomaly_component:.1f}/50 (raw score: {anomaly_score:.3f})")
        print(f"Cluster Component: {cluster_component:.1f}/50")
        if feature_adjustment != 0:
            print(f"Feature-Based Adjustment: {feature_adjustment:+.1f}")
        print(f"Cognitive Decline Risk Score: {risk_score:.1f}/100")
        
        if risk_score >= 80: risk_category = "Very High Risk"
        elif risk_score >= 65: risk_category = "High Risk"
        elif risk_score >= 45: risk_category = "Moderate Risk"
        elif risk_score >= 25: risk_category = "Low Risk"
        else: risk_category = "Minimal Risk"
        final_results['risk_category'] = risk_category
        print(f"Risk Category: {risk_category}")
    else:
         final_results['risk_score'] = None
         final_results['risk_category'] = "Error Processing"

    return final_results
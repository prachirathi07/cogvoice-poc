
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) 

import streamlit as st
import os
import whisper
import pickle
import tempfile 
import numpy as np
import pandas as pd
import time 

import config 
from prediction import get_prediction 


st.set_page_config(page_title="MemoTag Voice Analysis Task", page_icon="🧠", layout="wide")


def display_risk(risk_score, risk_category):
    """Displays the risk score and category with color coding."""
    if risk_score is None or risk_category is None or risk_category == "Error Processing":
        st.error("Could not determine risk category due to processing error.")
        return

    st.subheader("Risk Assessment")
    score_color = "grey"
   
    if risk_category == "High Risk":
        score_color = "red"
        st.error(f"**Risk Category: {risk_category}** ⚠️")
    elif risk_category == "Moderate Risk":
        score_color = "orange"
        st.warning(f"**Risk Category: {risk_category}**")
    elif risk_category == "Low/Minimal Risk":
        score_color = "green"
        st.success(f"**Risk Category: {risk_category}**")
    else: 
         st.info(f"**Risk Category: {risk_category}**")


    # Display score using Markdown for color
    st.markdown(f"""
    <div style="
        border: 2px solid {score_color};
        border-radius: 5px;
        padding: 15px;
        text-align: center;
        background-color: #f0f2f6;
    ">
        <h3 style='color:{score_color}; margin-bottom: 5px;'>Experimental Risk Score</h3>
        <p style='font-size: 2.5em; font-weight: bold; color:{score_color}; margin:0;'>
            {risk_score:.0f}/100
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Based on anomaly detection and cluster similarity to training data.")


@st.cache_resource 
def load_whisper_model():
    print("UI: Loading Whisper model (cached)...")
    try:
        model = whisper.load_model("base") 
        print("UI: Whisper model loaded.")
        return model
    except Exception as e:
        st.error(f"Fatal Error loading Whisper: {e}")
        return None


whisper_model_instance = load_whisper_model()
load_success = whisper_model_instance is not None 


# --- Sidebar ---
with st.sidebar:
   
    st.markdown("🧠 **MemoTag**", unsafe_allow_html=True) 
    st.header("Upload Audio")
    uploaded_file = st.file_uploader(
        "Choose a file (WAV, MP3, etc.)",
        type=['wav', 'mp3'],
        help="Upload spoken audio for analysis."
    )
    st.divider()
    st.info("🧠 **PoC Version**\n\nExperimental voice analysis.", icon="ℹ️")
    st.warning("""
    **Disclaimer:** This is a Proof-of-Concept and **NOT** a medical diagnosis tool.
    Results are experimental and based on limited data. Consult a healthcare professional.
    """, icon="⚠️")


# --- Main Area ---
st.title("MemoTag Voice Analysis Task")
st.markdown("Analyze speech patterns for *experimental* cognitive assessment.")
st.divider()

# --- Workflow ---
if not load_success:
    st.error("Application cannot proceed because the Whisper model failed to load. Please check console/logs.")
elif uploaded_file is not None:
    st.subheader("Uploaded Audio File")
    # Display audio player
    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes) 
   
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_audio_path = os.path.join(tmpdir, uploaded_file.name)
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        print(f"UI: Saved uploaded file temporarily to {temp_audio_path}")

        st.info("Processing... Analysis may take a few moments.", icon="⏳")
        analysis_placeholder = st.empty()

        prediction_results = None
        try:
            # Run Prediction
            with st.spinner('Performing feature extraction and analysis...'):
               
                prediction_results = get_prediction(
                    audio_path=temp_audio_path,
                    models_path=config.MODELS_PATH,
                    whisper_model_instance=whisper_model_instance
                )
               

            analysis_placeholder.success("Analysis Complete!", icon="✅")
            time.sleep(1)
            analysis_placeholder.empty() 

        except Exception as e:
            analysis_placeholder.error(f"An unexpected error occurred during analysis: {e}", icon="🚨")
            import traceback
            print(traceback.format_exc())

        finally:
             print(f"UI: Exited temporary directory scope for {temp_audio_path}")


    if prediction_results:
        if prediction_results.get("error"):
            st.error(f"Analysis Error: {prediction_results['error']}", icon="❌")
        else:
            display_risk(
                prediction_results.get('risk_score'),
                prediction_results.get('risk_category')
            )
            st.divider()

            tab1, tab2, tab3 = st.tabs(["📊 Extracted Features", "💬 Transcript", "⚙️ Model Details"])

            with tab1:
                st.subheader("Extracted Features")
                features = prediction_results.get('extracted_features', {})
                if features:
                    features_df_display = pd.DataFrame([features]).T
                    features_df_display.columns = ["Value"]
                    st.dataframe(features_df_display.round(3), use_container_width=True)
                else:
                    st.write("Could not display features.")

            with tab2:
                st.subheader("Transcript")
                st.markdown(f"> {prediction_results.get('transcript', 'N/A')}")

            with tab3:
                st.subheader("Model Prediction Details")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    pred_label = "Anomaly" if prediction_results.get('anomaly_prediction') == -1 else "Normal"
                    st.metric("Anomaly Prediction", pred_label)
                with col_m2:
                    score = prediction_results.get('anomaly_score', np.nan)
                    st.metric("Anomaly Score", f"{score:.4f}" if not np.isnan(score) else "N/A")
                with col_m3:
                     cluster = prediction_results.get('kmeans_cluster')
                     st.metric("Predicted Cluster", cluster if cluster is not None else "N/A")

                st.caption(f"Lower anomaly score = more anomalous.") 


    else:
        if not load_success:
             st.warning("Cannot show results as initial model loading failed.")

elif load_success:
    st.info("Upload an audio file using the sidebar to begin analysis.", icon="☝️")


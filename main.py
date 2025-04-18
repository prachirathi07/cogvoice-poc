import shutil
import os
import whisper 
from fastapi import FastAPI, File, UploadFile, HTTPException
import logging
import math
import uuid

from prediction import get_prediction
import config

whisper_model = None

def replace_nan_with_none(obj):
    """ Recursively replace NaN values in dicts/lists with None (JSON compatible). """
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(elem) for elem in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "temp_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="MemoTag Cognitive Assessment API",
    description="API for predicting cognitive decline risk from voice.",
    version="1.0.1" 
)


@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the MemoTag API!", "status": "OK"}

@app.post("/predict/")
async def create_prediction_request(uploaded_file: UploadFile = File(..., description="Audio file for analysis")):
    """
    Accepts audio file, loads model if needed, runs prediction, cleans results, returns JSON.
    """
    global whisper_model 

    if whisper_model is None:
        try:
            logger.info("Whisper model not loaded, loading now (first request)...")
            whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully for this instance.")
        except Exception as e:
             logger.error(f"CRITICAL Error: Could not load Whisper model on demand: {e}", exc_info=True)
             raise HTTPException(status_code=503, detail="Service Unavailable: Critical model loading failed.")

    if not uploaded_file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_extensions = {".wav", ".mp3"}
    file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
    if file_ext not in allowed_extensions:
         raise HTTPException(status_code=400, detail=f"Invalid file type '{file_ext}'. Allowed types: {', '.join(allowed_extensions)}")

    _, file_extension = os.path.splitext(uploaded_file.filename)
    temp_file_path = os.path.join(UPLOAD_DIR, f"upload_{os.urandom(8).hex()}{file_extension}")
    logger.info(f"Received file: {uploaded_file.filename}. Saving temporarily to {temp_file_path}")
    try:
        with open(temp_file_path, "wb") as file_object:
            shutil.copyfileobj(uploaded_file.file, file_object)
    except Exception as e:
        logger.error(f"Failed to save uploaded file {uploaded_file.filename} to {temp_file_path}: {e}", exc_info=True)
        await uploaded_file.close()
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
         await uploaded_file.close()

    prediction_results = None
    try:
        logger.info(f"Calling prediction pipeline for: {temp_file_path}")
        prediction_results = get_prediction(
            audio_path=temp_file_path,
            models_path=config.MODELS_PATH,
            whisper_model_instance=whisper_model
        )
        logger.info(f"Prediction completed for: {uploaded_file.filename}")

        if prediction_results and prediction_results.get("error"):
            logger.error(f"Prediction pipeline reported an error for {uploaded_file.filename}: {prediction_results['error']}")
            safe_results = replace_nan_with_none(prediction_results) # Clean even if error
            raise HTTPException(status_code=500, detail=safe_results.get("error", "Prediction Error"))

        logger.info("Cleaning prediction results for JSON compatibility...")
        safe_results = replace_nan_with_none(prediction_results)
        logger.info("Returning results.")
        return safe_results

    except HTTPException as e:
        logger.error(f"HTTP Exception occurred for {uploaded_file.filename}: Status={e.status_code}, Detail={e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during prediction processing for {uploaded_file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error occurred during prediction processing: {e}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Successfully removed temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file {temp_file_path}: {e}")
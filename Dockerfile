

# 1. Start with an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (ffmpeg for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY the requirements file first (better caching)
COPY requirements.txt .

# 5. Install Python dependencies from the requirements file
# Ensure torch is included in requirements.txt for Whisper
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Set NLTK_DATA environment variable (standard location inside container)
ENV NLTK_DATA=/usr/share/nltk_data

# 7. Download NLTK resources during build, using NLTK_DATA path
RUN python -c "import nltk; import os; os.makedirs(os.environ['NLTK_DATA'], exist_ok=True); \
    nltk.download('popular', download_dir=os.environ['NLTK_DATA']); \
    nltk.download('punkt', download_dir=os.environ['NLTK_DATA']); \
    nltk.download('all', download_dir=os.environ['NLTK_DATA'])"

# 8. Copy the rest of your application code and data
# Assumes data/models/models.pkl exists relative to this Dockerfile
COPY . .

# 9. Pre-download Whisper model during build (optional but recommended)
# Change 'base' if your main.py uses a different size
RUN python -c "import whisper; whisper.load_model('base')"

# 10. Expose the port the app runs on
EXPOSE 8000

# 11. Define the command to run your application (main.py with app object)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
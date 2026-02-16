import os
import io
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import torch
from dotenv import load_dotenv

from config import MODELS_DIR, PROCESSED_DATA_DIR
from ml.generate import load_model, generate_solo_from_model

load_dotenv()


# Load model ONCE at startup
print("Loading model...")
MODEL_NAME = "best_model_v3.pt"
VOCAB_FILE_NAME = "vocab_v3.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = MODELS_DIR / "jazz_transformer" / MODEL_NAME
vocab_path = PROCESSED_DATA_DIR / VOCAB_FILE_NAME

MODEL, VOCAB, SOS_ID, EOS_ID = load_model(checkpoint_path, vocab_path, DEVICE)
print(f"Model loaded on {DEVICE}!")


# Create Flask app
def create_app():
    app = Flask(__name__)
    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    CORS(app, origins=allowed_origins)


    # Health endpoint to check if server is running and model is loaded
    @app.get("/api/health")
    def health():
        return {"status": "ok", "device": str(DEVICE)}


    # Generate endpoint that returns generated MIDI file
    @app.post("/api/generate")
    def generate():
        print("Received request for solo generation")
        beat_chord_data = request.json
        try:
            midi_buffer = generate_solo_from_model(
                beat_chord_data,
                MODEL, VOCAB, SOS_ID, EOS_ID, DEVICE
            )
            return send_file(
                midi_buffer,
                mimetype='audio/midi',
                as_attachment=True,
                download_name=f"{beat_chord_data.get('solo_info', {}).get('outputFileName', 'jazz_solo')}.mid"
            )
        except Exception as e:
            print(f"Error during solo generation: {e}")
            return jsonify({"error": str(e)}), 500

    return app

app = create_app()
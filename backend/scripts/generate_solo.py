# ml/generate.py

import os
import sys
from pathlib import Path
import tempfile
import io

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import json
import torch
from ml.models.transformer import JazzTransformer
from ml.preprocessing.tokenizer import Tokenizer
from utils.post_processor import post_process_solo

MODEL_NAME = "best_model_v3.pt"
VOCAB_FILE_NAME = "vocab_v3.json"

def load_model(checkpoint_path, vocab_path, device):
    """Load trained model from checkpoint."""
    
    # Load vocab
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    vocab_size = len(vocab)
    pad_id = vocab["<PAD>"]
    sos_id = vocab["<SOS>"]
    eos_id = vocab["<EOS>"]
    
    # Create model
    model = JazzTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.0,  # No dropout during inference
        max_encoder_len=700,   # Match training config
        max_decoder_len=4000,  # Match training config
        pad_id=pad_id
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, vocab, sos_id, eos_id


def _generate_tokens(model, vocab, encoder_tokens, device, sos_id, eos_id, 
                  temperature=1.0, top_k=50, top_p=0.95, max_len=2000):
    """Generate a jazz solo given chord progression."""
    
    # Convert tokens to ids
    print("Converting tokens to IDs...")
    encoder_ids = [vocab.get(t, vocab["<PAD>"]) for t in encoder_tokens]
    src = torch.tensor([encoder_ids], dtype=torch.long, device=device)
    
    # Generate
    print("Generating solo tokens...")
    generated_ids = model.generate(
        src,
        max_len=max_len,
        sos_id=sos_id,
        eos_id=eos_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    
    # Convert back to tokens
    id_to_token = {v: k for k, v in vocab.items()}
    generated_tokens = [id_to_token.get(i, "<UNK>") for i in generated_ids]
    
    return generated_tokens


def generate_solo_from_model(beat_chord_data, model, vocab, sos_id, eos_id, device):
    """Generate solo using pre-loaded model, return in-memory buffer."""
    
    tokenizer = Tokenizer(vocab)
    
    input_tokens = tokenizer.encode_chord_timeline(
        beat_chord_data['beats'], 
        beat_chord_data['solo_info']
    )
    
    print("Generating solo...")
    generated = _generate_tokens(
        model, vocab, input_tokens, device, sos_id, eos_id,
        temperature=0.9, top_k=50, top_p=0.95
    )
    
    print("Generated tokens:", len(generated))
    
    events = tokenizer.decode_melody(generated)
    processed_events = post_process_solo(events, beat_chord_data['beats'])
    
    # Create MIDI in memory
    from midiutil import MIDIFile
    midi_file = MIDIFile(1)
    tempo = beat_chord_data['solo_info']['avgtempo']
    midi_file.addTrackName(0, 0, "Generated Solo")
    midi_file.addTempo(0, 0, tempo)
    
    for event in processed_events:
        midi_file.addNote(0, 0, event['pitch'], event['onset'], event['duration'], 100)
    
    # Write to buffer instead of file
    buffer = io.BytesIO()
    midi_file.writeFile(buffer)
    buffer.seek(0)
    
    return buffer

def main():
    # for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    checkpoint_path = BACKEND_ROOT / "models" / "jazz_transformer" / MODEL_NAME
    vocab_path = BACKEND_ROOT / "data" / "processed" / VOCAB_FILE_NAME
    
    # Load model
    print("Loading model...")
    model, vocab, sos_id, eos_id = load_model(checkpoint_path, vocab_path, device)

    # Init tokenizer
    tokenizer = Tokenizer(vocab)

    # Create pandas DataFrame for beats and solo_info for testing
    import pandas as pd
    solo_info = {
        'key': 'Eb_MAJOR',
        'style': 'COOL',
        'avgtempo': 120
    }

    # D_MIN7 G_DOM C_MAJ7 Db_DIM7 repeated for 8 times (reformat to json )
    beats = [
        {'bar': 0, 'beat': 1, 'root': 'F', 'quality_class': 'MIN7'},
        {'bar': 1, 'beat': 1, 'root': 'Bb', 'quality_class': 'DOM'},
        {'bar': 2, 'beat': 1, 'root': 'Eb', 'quality_class': 'MAJ7'},
        {'bar': 3, 'beat': 1, 'root': 'Ab', 'quality_class': 'MAJ7'},
        {'bar': 4, 'beat': 1, 'root': 'D', 'quality_class': 'HDIM'},
        {'bar': 5, 'beat': 1, 'root': 'G', 'quality_class': 'AUG'},
        {'bar': 6, 'beat': 1, 'root': 'C', 'quality_class': 'MIN7'},
        {'bar': 7, 'beat': 1, 'root': 'C', 'quality_class': 'DOM'},
        {'bar': 8, 'beat': 1, 'root': 'F', 'quality_class': 'MIN7'},
        {'bar': 9, 'beat': 1, 'root': 'Bb', 'quality_class': 'DOM'},
        {'bar': 10, 'beat': 1, 'root': 'Eb', 'quality_class': 'MAJ7'},
        {'bar': 11, 'beat': 1, 'root': 'Ab', 'quality_class': 'MAJ7'},
        {'bar': 12, 'beat': 1, 'root': 'D', 'quality_class': 'HDIM'},
        {'bar': 13, 'beat': 1, 'root': 'G', 'quality_class': 'AUG'},
        {'bar': 14, 'beat': 1, 'root': 'C', 'quality_class': 'MIN7'},
        {'bar': 15, 'beat': 1, 'root': 'C', 'quality_class': 'DOM'},
        {'bar': 16, 'beat': 1, 'root': 'F', 'quality_class': 'MIN7'},
        {'bar': 17, 'beat': 1, 'root': 'Bb', 'quality_class': 'DOM'},
        {'bar': 18, 'beat': 1, 'root': 'Eb', 'quality_class': 'MAJ7'},
        {'bar': 19, 'beat': 1, 'root': 'Ab', 'quality_class': 'MAJ7'},
        {'bar': 20, 'beat': 1, 'root': 'D', 'quality_class': 'HDIM'},
        {'bar': 21, 'beat': 1, 'root': 'G', 'quality_class': 'AUG'},
        {'bar': 22, 'beat': 1, 'root': 'C', 'quality_class': 'MIN7'},
        {'bar': 23, 'beat': 1, 'root': 'C', 'quality_class': 'DOM'},
        {'bar': 24, 'beat': 1, 'root': 'F', 'quality_class': 'MIN7'},
        {'bar': 25, 'beat': 1, 'root': 'Bb', 'quality_class': 'DOM'},
        {'bar': 26, 'beat': 1, 'root': 'Eb', 'quality_class': 'MAJ7'},
        {'bar': 27, 'beat': 1, 'root': 'Ab', 'quality_class': 'MAJ7'},
        {'bar': 28, 'beat': 1, 'root': 'D', 'quality_class': 'HDIM'},
        {'bar': 29, 'beat': 1, 'root': 'G', 'quality_class': 'AUG'},
        {'bar': 30, 'beat': 1, 'root': 'C', 'quality_class': 'MIN7'},
        {'bar': 31, 'beat': 1, 'root': 'C', 'quality_class': 'DOM'},

    ]

    # Encode chord timeline from beat_chord_data
    input_tokens = tokenizer.encode_chord_timeline(beats, solo_info)
    print("ENCODER TOKENS:", input_tokens)

    # Check which tokens are unknown
    for tok in input_tokens:
        if tok not in vocab:
            print(f"UNKNOWN TOKEN: {tok}")
    
    print("Generating solo...")
    generated = _generate_tokens(
        model, vocab, input_tokens, device, sos_id, eos_id,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )
    
    print("\nGenerated tokens:")
    print(generated)

    # function to make midi file from events
    def midi_from_events(events, solo_info, file_name="generated_solo.mid"):

        # make midi file from events
        from midiutil import MIDIFile
        midi_file = MIDIFile(1)
        track = 0
        time = 0
        tempo = solo_info['avgtempo'] 
        volume = 100   
        midi_file.addTrackName(track, time, "Generated Solo")
        midi_file.addTempo(track, time, tempo)
        for event in events:
            onset = event['onset']
            pitch = event['pitch']
            duration = event['duration']
            midi_file.addNote(track, 0, pitch, onset, duration, volume)


        # Save MIDI file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            output_path = Path(f.name)
            midi_file.writeFile(f)

        return output_path

    # without post-processing
    events = tokenizer.decode_melody(generated)
    midi_from_events(events, solo_info, file_name="generated_solo_no_post.mid")
    # with post-processing
    processed_events = post_process_solo(events, beats)
    midi_from_events(processed_events, solo_info, file_name="generated_solo_post.mid")

if __name__ == "__main__":
    main()
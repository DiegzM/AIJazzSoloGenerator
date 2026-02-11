import pandas as pd
import sys
from pathlib import Path
from midiutil import MIDIFile
import pickle
import json

import ml.preprocessing.loader as loader
import ml.preprocessing.cleaner as cleaner
import ml.preprocessing.feature_extractor as feature_extractor
from ml.preprocessing.tokenizer import Tokenizer

from config import PROCESSED_DATA_DIR

VOCAB_FILE_NAME = "vocab_v3.json"

if __name__ == "__main__":

    # Load, clean, extract features
    data = loader.load()
    data = cleaner.clean(data)
    features = feature_extractor.extract_features(data)

    # Tokenize features

    all_sequences = []

    tokenizer = Tokenizer()

    for melid in features['solo_info']['melid'].unique():
        melid_beats = features['beats'][features['beats']['melid'] == melid]
        melid_melody = features['melody'][features['melody']['melid'] == melid]
        melid_solo_info = features['solo_info'][features['solo_info']['melid'] == melid]
        
        tokens = tokenizer.encode_solo(melid_melody, melid_beats, melid_solo_info)

        if tokens is not None:
            all_sequences.append((tokens['encoder_input'], tokens['decoder_target']))

        print(f"Melody ID: {melid}")

    print(f"Total tokenized sequences: {len(all_sequences)}")

    tokenizer.build_vocab_from_sequences(all_sequences)

    # Save tokenizer vocabulary
    vocab_path = PROCESSED_DATA_DIR / VOCAB_FILE_NAME
    with open(vocab_path, 'w') as f:
        json.dump(tokenizer.vocab, f)

    print(f"\nTokenizer vocabulary saved to {vocab_path}")

    # Save tokenized sequences
    token_pairs = []

    for enc_tokens, dec_tokens in all_sequences:
        enc_ids = [tokenizer.token_to_id(tok) for tok in enc_tokens]
        dec_ids = [tokenizer.token_to_id(tok) for tok in dec_tokens]
        token_pairs.append((enc_ids, dec_ids))

    pairs_path = PROCESSED_DATA_DIR / "token_pairs.pkl"
    with open(pairs_path, "wb") as f:
        pickle.dump(token_pairs, f)

    print(f"Tokenized sequences saved to {pairs_path}")
    
    print("Preprocessing completed successfully.")
    print("")


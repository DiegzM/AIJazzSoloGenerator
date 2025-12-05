import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / 'backend'
sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd
from midiutil import MIDIFile

import ml.preprocessing.loader as loader
import ml.preprocessing.cleaner as cleaner
import ml.preprocessing.feature_extractor as feature_extractor
from ml.preprocessing.tokenizer import Tokenizer
from utils.chord_parsers import WeimarChord

data = loader.load()
data = cleaner.clean(data)
features = feature_extractor.extract_features(data)
tokenizer = Tokenizer(features)

def create_midi(melid, export=False):

    try:
        melid_melody = features['melody'][features['melody']['melid'] == melid]
        melid_beats = features['beats'][features['beats']['melid'] == melid]
        melid_solo_info = features['solo_info'][features['solo_info']['melid'] == melid]

        melid_melody_tokens = tokenizer.encode_solo(
            melid_melody,
            melid_beats,
            melid_solo_info
        )

        if melid_melody_tokens is None:
            print(f"No tokens generated for melid {melid}. Skipping MIDI creation.")
            return

        melid_melody_events = tokenizer.decode_melody(melid_melody_tokens["decoder_target"])

        # Create track
        midi_file = MIDIFile(3)

        # Init track
        track = 0
        time = 0
        tempo = float(melid_solo_info['avgtempo'].iloc[0])
        volume = 100

        midi_file.addTrackName(track, time, f"Melody {melid}")
        midi_file.addTempo(track, time, tempo)

        # generate midi using datasets['melody'] and datasets['beats']

        # For event in melid_melody_events, add note to midi
        for event in melid_melody_events:
            onset = event['onset']
            pitch = event['pitch']
            duration = event['duration']
            midi_file.addNote(track, 0, pitch, onset, duration, volume)

        # Init another track for chords
        chord_track = 1
        midi_file.addTrackName(chord_track, time, f"Chords {melid}")

        for index, row in melid_beats.iterrows():
            onset = row['onset']
            chord: WeimarChord = row['chord']
            # Get chord pitches
            chord_pitches = chord.get_chord_pitches()
            duration = 1  # Assume each chord lasts 1 beat
            for pitch in chord_pitches:
                midi_file.addNote(chord_track, 0, pitch, onset, duration, volume)

        # Init another track for bass
        bass_track = 2
        midi_file.addTrackName(bass_track, time, f"Bass {melid}")
        for index, row in melid_beats.iterrows():
            onset = row['onset']
            bass_pitch = row['bass_pitch']
            if pd.isna(bass_pitch):
                continue
            pitch = int(bass_pitch)
            duration = 1  # Assume each bass note lasts 1 beat
            midi_file.addNote(bass_track, 0, pitch, onset, duration, volume)


        # Write to projectroot/data/processed/test_midi.mid
        if export:
            with open(PROJECT_ROOT / 'data' / 'processed' / 'test_midi.mid', 'wb') as output_file:
                midi_file.writeFile(output_file)
        
        midi_file.close()
        print(f"MIDI file for melid {melid} created successfully.")
    except Exception as e:
        print(f"Error creating MIDI for melid {melid}: {e}")


export_midi = 223
for melid in features['melody']['melid'].unique():
    create_midi(melid, export=(melid == export_midi))
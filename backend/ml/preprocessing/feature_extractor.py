import pandas as pd
import numpy as np
from rich.console import Console
from config.settings import RAW_DATA_DIR

from utils.chord_parsers import WeimarChord, convert_metadata_key

console = Console()

# Convert onsets to beat grid using beats dataset
def onsets_to_beats(dataset):
    console.print("\nConverting onsets to beat grid using beats dataset...")
    beats = dataset['beats']
    melody = dataset['melody']

    melody = melody.copy()
    beats = beats.copy()

    # For each melid convert onsets to beat based
    for melid in melody['melid'].unique():
        melid_beats = beats[beats['melid'] == melid]
        melid_melody = melody[melody['melid'] == melid]

        if melid_beats.empty or melid_melody.empty:
            continue

        beat_onsets = melid_beats['onset'].values
        beat_numbers = np.arange(len(beat_onsets))

        mapped_beats = np.interp(
            melid_melody['onset'].values,
            beat_onsets,
            beat_numbers
        )

        melody.loc[melody['melid'] == melid, 'onset'] = mapped_beats
        beats.loc[beats['melid'] == melid, 'onset'] = beat_numbers

    dataset['melody'] = melody
    dataset['beats'] = beats

    return dataset

# Convert duration to beats using solo_info avgtempo
def duration_to_beats(dataset):
    console.print("\nConverting onsets to beat grid using beats dataset...")
    melody = dataset['melody'].copy()
    solo_info = dataset['solo_info']

    for melid in solo_info['melid']:
        melid_melody = melody[melody['melid'] == melid]
        melid_solo_info = solo_info[solo_info['melid'] == melid]
        duration = melid_melody['duration']
        tempo = float(melid_solo_info['avgtempo'].iloc[0])
        bps = tempo / 60

        mapped_durations = melid_melody['duration'].map(lambda x: x * bps)
        melody.loc[melody['melid'] == melid, 'duration'] = mapped_durations

    dataset['melody'] = melody
    return dataset

# Quantize onsets and durations to nearest 1/24 beat
def quantize_to_nearest_24th(dataset) -> str:
    console.print("\nQuantizing onsets and durations to nearest 1/24 beat...")
    melody = dataset['melody']
    melody['onset'] = melody['onset'].apply(lambda x: round(x * 24) / 24)
    melody['duration'] = melody['duration'].apply(lambda x: max(round(x * 24) / 24, 1/24))
    dataset['melody'] = melody
    return dataset

# Remove duplicated notes in melody (same melid, onset, pitch) (keep the one with longer duration)
def remove_duplicated_notes(dataset):
    console.print("\nRemoving duplicated notes in melody...")

    melody = dataset['melody']
    melody = melody.sort_values(
        ['melid', 'onset', 'pitch', 'duration'],
        ascending=[True, True, True, False]
    )
    melody = melody.drop_duplicates(
        subset=['melid', 'onset', 'pitch'],
        keep='first'
    )
    melody = melody.sort_values(
        ['melid', 'onset', 'pitch']
    ).reset_index(drop=True)
    dataset['melody'] = melody

    return dataset

# Replace the 'chord' column with their weimarchord instance
def replace_weimar_chord(dataset) -> str:
    console.print("\nReplacing 'chord' column with WeimarChord instances...")
    beats = dataset['beats']
    beats['chord'] = beats['chord'].apply(lambda x: WeimarChord(x).convert_to_enharmonic_flat())
    dataset['beats'] = beats
    return dataset

# Add root, quality, quality_class, extensions and bass columns to beats dataset
def add_chord_features(dataset) -> str:
    console.print("\nAdding chord feature columns to beats dataset...")
    beats = dataset['beats']
    beats['root'] = beats['chord'].apply(lambda x: x.root)
    beats['quality'] = beats['chord'].apply(lambda x: x.quality)
    beats['quality_class'] = beats['chord'].apply(lambda x: x.quality_class)
    beats['extensions'] = beats['chord'].apply(lambda x: x.extensions)
    beats['bass'] = beats['chord'].apply(lambda x: x.bass)
    dataset['beats'] = beats
    return dataset

# Replace metadata key to a standard enharmonic flat notation in solo_info and beats datasets
def replace_metadata_key_chords(dataset) -> str:
    console.print("\nReplacing metadata key chords to standard enharmonic flat notation...")
    solo_info = dataset['solo_info']

    solo_info['key'] = solo_info['key'].apply(lambda x: convert_metadata_key(x))

    dataset['solo_info'] = solo_info
    return dataset

# Round tempos to nearest 10
def round_tempos(dataset):
    console.print("\nRounding tempos to nearest 10 v...")
    solo_info = dataset['solo_info']
    solo_info['avgtempo'] = solo_info['avgtempo'].round(-1)
    solo_info['avgtempo'] = solo_info['avgtempo'].apply(lambda x: max(x, 40))
    dataset['solo_info'] = solo_info
    return dataset

def extract_features(dataset):
    console.print("\nExtracting features from dataset...")
    dataset = onsets_to_beats(dataset)
    dataset = duration_to_beats(dataset)
    dataset = quantize_to_nearest_24th(dataset)
    dataset = remove_duplicated_notes(dataset)
    dataset = replace_weimar_chord(dataset)
    dataset = replace_metadata_key_chords(dataset)
    dataset = add_chord_features(dataset)
    dataset = round_tempos(dataset)
    return dataset
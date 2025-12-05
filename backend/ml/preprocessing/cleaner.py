import pandas as pd
from rich.console import Console
from config.settings import RAW_DATA_DIR

console = Console()

# Required columns for each dataset
REQUIRED_COLUMNS = {
    "melody": ["melid", "onset", "pitch", "duration", "bar"],
    "beats": ["melid", "onset", "bar", "beat", "chord", "bass_pitch"],
    "sections": ["melid", "type", "start", "end", "value"],
    "solo_info": ["melid", "avgtempo", "key", "style", "signature"],
}

# Extract only requied columns from datasets
def _extract_required_columns(datasets):
    console.print("\nExtracting required columns from datasets...")
    for dataset_key, required_cols in REQUIRED_COLUMNS.items():
        if dataset_key in datasets:
            datasets[dataset_key] = datasets[dataset_key][required_cols]
    return datasets

# Drop melid rows where solo_info 'key' is NaN and bass pitch is nan for all rows in a melid
def _drop_melid_rows(datasets):

    # Get rows in solo_info where key is NaN
    
    solo_info = datasets['solo_info']
    nan_key_rows = solo_info[solo_info['key'].isna()]
    melids_with_nan_key = nan_key_rows['melid'].unique()

    console.print(f"[yellow]Dropping melodies with NaN key in solo_info: {str(melids_with_nan_key)}[/yellow]")

    # Drop melodies with NaN key from all datasets
    for dataset_key, df in datasets.items():
        datasets[dataset_key] = df[~df['melid'].isin(melids_with_nan_key)].reset_index(drop=True)

    # Delete melids that arent 4/4 time signature from all datasets
    melids_to_drop = solo_info[solo_info['signature'] != '4/4']['melid'].unique()
    console.print(f"[yellow]Dropping melodies with non 4/4 time signature: {str(melids_to_drop)}[/yellow]")
    for dataset_key, df in datasets.items():
        datasets[dataset_key] = df[~df['melid'].isin(melids_to_drop)].reset_index(drop=True)

    return datasets

# sections rows only where "type" is "CHORD"
def _clean_sections(datasets):
    console.print("\nCleaning sections dataset...")
    sections = datasets['sections']
    chord_sections = sections[sections['type'] == 'CHORD'].reset_index(drop=True)
    datasets['sections'] = chord_sections
    return datasets

# Shift onsets so that melid's beats first beat event starts at onset 0
def _normalize_onsets(datasets):
    beats = datasets['beats']
    melody = datasets['melody']

    console.print("\nNormalizing onsets in beats and melody datasets...")
    for melid in beats['melid'].unique():
        melid_beats = beats[beats['melid'] == melid]
        melid_melody = melody[melody['melid'] == melid]

        first_beat_onset = melid_beats[melid_beats['beat'] == 1]['onset'].min()

        # Shift onsets such that first beat onset is 0
        beats.loc[beats['melid'] == melid, 'onset'] -= first_beat_onset
        melody.loc[melody['melid'] == melid, 'onset'] -= first_beat_onset

    datasets['beats'] = beats
    datasets['melody'] = melody

    return datasets

# Shift bar such that the first bar starts at 0
def _normalize_bars(datasets):
    beats = datasets['beats']
    melody = datasets['melody']

    console.print("\nNormalizing bars in beats and melody datasets...")
    for melid in beats['melid'].unique():
        melid_beats = beats[beats['melid'] == melid]
        melid_melody = melody[melody['melid'] == melid]

        first_bar = min(melid_beats['bar'].min(), melid_melody['bar'].min())

        # Shift bars such that first bar is 0
        beats.loc[beats['melid'] == melid, 'bar'] -= first_bar
        melody.loc[melody['melid'] == melid, 'bar'] -= first_bar

    datasets['beats'] = beats
    datasets['melody'] = melody

    return datasets

# Fill missing chord labels in beats dataset (forward fill) PER melid
def _fill_chords(datasets):
    console.print("\nFilling missing chord labels in beats dataset...")
    beats = datasets['beats']
    beats['chord'] = beats.groupby('melid')['chord'].ffill().bfill()
    datasets['beats'] = beats
    return datasets

def clean(datasets):
    console.print("\nCleaning datasets...")
    datasets = _extract_required_columns(datasets)
    datasets = _drop_melid_rows(datasets)
    datasets = _clean_sections(datasets)
    datasets = _normalize_onsets(datasets)
    datasets = _normalize_bars(datasets)
    datasets = _fill_chords(datasets)
    return datasets
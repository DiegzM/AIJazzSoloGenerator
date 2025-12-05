# Example of beats row: {'bar': 0, 'beat': 1, 'root': 'B', 'quality_class': 'MAJ7'},
# Example of solo row: {'onset': 0.0, 'duration': 0.5, 'pitch': 60}
# Onset and duration are in beats

import os
import sys
from pathlib import Path
import random

UTILS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(UTILS_ROOT))


from chord_parsers import get_chord_scale

# Post-process the solo to fit chord scales
def post_process_solo(solo, beats):
    """Post-process the generated solo to be more musically coherent."""
    print("Post-processing solo...")
    print("Getting chord segments from beats...")
    chord_segments = _build_chord_segments(beats)

    print("Computing scales for each chord segment...")
    chord_segments = _get_scales_for_chord(chord_segments)

    print("Adjusting solo notes to fit chord scales...")
    processed_solo = []
    for note in solo:
        processed_note = _process_note(note['onset'], note['duration'], note['pitch'], chord_segments)
        processed_solo.append(processed_note)

    # Fix overlaps
    print("Fixing overlapping notes...")
    processed_solo = _fix_overlaps(processed_solo)
    
    return processed_solo


# Build chord segments from beats
def _build_chord_segments(beats):
    segments = []

    # Convert each beat entry to an absolute time
    # Skip beats with no chord (empty root)
    for b in beats:
        if not b["root"] or b["root"] == "":  # Skip "No Chord" beats
            continue
        start = b["bar"] * 4 + (b["beat"] - 1)
        segments.append({
            "start": start,
            "root": b["root"],
            "quality": b["quality_class"],
        })

    # If no chords at all, return empty
    if not segments:
        return []

    # Sort by start time
    segments.sort(key=lambda x: x["start"])

    # Add end times (next segment's start)
    for i in range(len(segments) - 1):
        segments[i]["end"] = segments[i+1]["start"]

    # Last chord extends to infinity
    segments[-1]["end"] = float("inf")

    return segments

# Get scales for each chord segment
def _get_scales_for_chord(segments):
    for segment in segments:
        segment['scale'] = get_chord_scale(segment['root'], segment['quality'])
    return segments


# Process individual note to fit chord scale
def _process_note(onset, duration, pitch, chord_segments):
    # If no chord segments, return note unchanged
    if not chord_segments:
        return {"onset": onset, "duration": duration, "pitch": pitch}
    
    # Get current chord segment
    current_segment = None
    for segment in chord_segments:
        if segment['start'] <= onset < segment['end']:
            current_segment = segment
            break
    if current_segment is None:
        return {"onset": onset, "duration": duration, "pitch": pitch}
    
    scale = current_segment['scale']

    # If no scale or pitch is in scale, return as is
    if not scale or pitch in scale:
        return {"onset": onset, "duration": duration, "pitch": pitch}

    # Find nearest note in the scale
    nearest = min(scale, key=lambda s: abs(s - pitch))

    # 50/50 tie-breaking for equidistant notes
    ties = [s for s in scale if abs(s - pitch) == abs(nearest - pitch)]
    if len(ties) > 1:
        nearest = random.choice(ties)

    return {"onset": onset, "duration": duration, "pitch": nearest}

def _fix_overlaps(solo, min_duration=0.05):
    """Ensure no notes overlap by shortening durations."""
    if not solo:
        return solo
    
    solo = sorted(solo, key=lambda n: n['onset'])  # just in case

    for i in range(len(solo) - 1):
        current = solo[i]
        next_note = solo[i + 1]

        end_time = current['onset'] + current['duration']

        # If overlap, clamp
        if end_time > next_note['onset']:
            new_duration = max(min_duration, next_note['onset'] - current['onset'])
            current['duration'] = new_duration

    return solo

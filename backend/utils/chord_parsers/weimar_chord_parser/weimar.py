from dataclasses import dataclass

from config.settings import RAW_DATA_DIR
from .normalization import _parse_chord
from .definitions import QUALITY_INTERVALS, EXTENSION_INTERVALS, PITCH_INTERVALS, PITCH_INTERVALS_ABSOLUTE, ENHARMONIC_FLAT_EQUIVALENTS, CHORD_SCALES

class WeimarChord:
    def __init__(self, label: str):
        self.label = label
        root, quality, quality_class, extensions, bass = _parse_chord(label)

        if root is None:  # Invalid chord label
            self.root = ""
            self.quality = ""
            self.quality_class = ""
            self.extensions = []
            self.bass = ""
            self.is_valid = False
            self.root_pitch = 0
            self.chord_intervals = []
        else:
            self.root = root
            self.quality = quality
            self.quality_class = quality_class
            self.extensions = extensions or []
            self.bass = bass or root
            self.is_valid = True
            self.root_pitch = self._get_root_pitch(root)
            self.chord_intervals = self._build_chord()


    def to_string(self) -> str:
        return f"{self.root} {self.quality} {self.quality_class} {' '.join(self.extensions)} {self.bass}"
    
    def get_info(self):
        return {
            "root": self.root,
            "quality": self.quality,
            "quality_class": self.quality_class,
            "extensions": self.extensions,
            "bass": self.bass
        }
    
    def get_chord_pitches(self) -> list[int]:
        return [self.root_pitch + interval for interval in self.chord_intervals]
    
    def convert_to_enharmonic_flat(self):
        if self.root in ENHARMONIC_FLAT_EQUIVALENTS:
            self.root = ENHARMONIC_FLAT_EQUIVALENTS[self.root]
            self.root_pitch = self._get_root_pitch(self.root)
        if self.bass in ENHARMONIC_FLAT_EQUIVALENTS:
            self.bass = ENHARMONIC_FLAT_EQUIVALENTS[self.bass]

        return self

    def _get_root_pitch(self, root: str) -> int: 
        return 60 + PITCH_INTERVALS.get(root, 0)

    def _build_chord(self):
        intervals = QUALITY_INTERVALS.get(self.quality, []).copy()

        for ext in self.extensions:
            interval = EXTENSION_INTERVALS.get(ext)
            if interval:
                if interval in ['b5', '#5'] and 7 in intervals:
                    intervals.remove(7)
                if interval not in intervals:
                    intervals.append(interval)

        return sorted(intervals)

# Function to convert metadata key (originally stored differently than beats chords)
# Ex: C-maj or C-min or Ab-chrom
def convert_metadata_key(key: str) -> str:

    # Get root (before "-") and replace with enharmonic flat if needed
    if "-" in key:
        parts = key.split("-", 1)
        root = parts[0]
        if root in ENHARMONIC_FLAT_EQUIVALENTS:
            root = ENHARMONIC_FLAT_EQUIVALENTS[root]
        key = root + "-" + parts[1]
    else:
        root = key
        if root in ENHARMONIC_FLAT_EQUIVALENTS:
            root = ENHARMONIC_FLAT_EQUIVALENTS[root]
        key = root

    # Replace suffixes with standard root notation
    if "maj" in key:
        root = key.replace("-maj", "_MAJOR")
        return root
    elif "min" in key:
        root = key.replace("-min", "_MINOR")
        return root
    elif "chrom" in key:
        root = key.replace("-chrom", "_MINOR")
        return root 
    elif "mix" in key:
        root = key.replace("-mix", "_MAJOR")
        return root
    elif "dor" in key:
        root = key.replace("-dor", "_MINOR")
        return root
    elif "blues" in key:
        root = key.replace("-blues", "_MAJOR")
        return root
    elif "lyd" in key:
        root = key.replace("-lyd", "_MAJOR")
        return root
    else:
        return key + "_MAJOR"  # Default to major if unknown
    
def convert_to_enharmonic_flat(self):
    if self.root in ENHARMONIC_FLAT_EQUIVALENTS:
        self.root = ENHARMONIC_FLAT_EQUIVALENTS[self.root]
        self.root_pitch = self._get_root_pitch(self.root)
    if self.bass in ENHARMONIC_FLAT_EQUIVALENTS:
        self.bass = ENHARMONIC_FLAT_EQUIVALENTS[self.bass]

    return self

# Get all the appropriate chord scale pitches given root and quality_class from 0-127
def get_chord_scale(root, quality_class):

    scale_intervals = CHORD_SCALES.get(quality_class, [])
    root_pitch = PITCH_INTERVALS_ABSOLUTE.get(root, 0)

    scale_notes = []
    for pitch in range(0, 128):
        pitch_class = (pitch - root_pitch) % 12
        if pitch_class in scale_intervals:
            scale_notes.append(pitch)

    for note in scale_notes:
        # convert to letter notation for debugging
        note_pc = note % 12
        for k, v in PITCH_INTERVALS_ABSOLUTE.items():
            if v % 12 == note_pc:
                break

    return scale_notes

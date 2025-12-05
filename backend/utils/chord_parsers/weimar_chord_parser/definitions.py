QUALITY_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "maj6": [0, 4, 7, 9],
    "min6": [0, 3, 7, 9],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "minmaj7": [0, 3, 7, 11],
    "hdim7": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
}

EXTENSION_INTERVALS = {
    "b5": 6,
    "#5": 8,
    "9": 14,
    "b9": 13,
    "#9": 15,
    "11": 17,
    "#11": 18,
    "b11": 16,
    "13": 21,
    "b13": 20,
    "#13": 22,
}

PITCH_INTERVALS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "E#": 5,
    "Fb": 4,
    "F": 5,
    "F#": 6,
    "Gb": -6,
    "G": -5,
    "G#": -4,
    "Ab": -4,
    "A": -3,
    "A#": -2,
    "Bb": -2,
    "B": -1,
    "B#": 0,
    "Cb": -1,
}

PITCH_INTERVALS_ABSOLUTE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "E#": 5,
    "Fb": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "B#": 12,
    "Cb": 11,
}




# Quality patterns for chord parsing
QUALITY_PATTERNS = {
    # Major triad qualities
    '': 'maj',
    'M': 'maj',
    'maj': 'maj',
    'major': 'maj',
    'j': 'maj',

    # Minor triad qualities
    'm': 'min',
    'min': 'min',
    'minor': 'min',
    '-': 'min',

    # Diminished triad qualities
    'dim': 'dim',
    'diminished': 'dim',
    'o': 'dim',

    # Augmented triad qualities
    'aug': 'aug',
    'augmented': 'aug',
    '+': 'aug',

    # Suspended triad qualities
    'sus2': 'sus2',
    'sus4': 'sus4',
    'sus': 'sus4',
    'suspended': 'sus4',

    # Dominant 7 chord qualities
    '7': '7',
    '9': '7',
    '11': '7',
    '13': '7',
    '7alt': '7',
    '9alt': '7',
    '11alt': '7',
    '13alt': '7',
    'dom7': '7',
    'dom': '7',
    'dominant': '7',
    'dominant7': '7',

    # Major 7 chord qualities
    'maj7': 'maj7',
    'maj9': 'maj7',
    'maj11': 'maj7',
    'maj13': 'maj7',
    'M7': 'maj7',
    'M9': 'maj7',
    'M11': 'maj7',
    'M13': 'maj7',
    'major7': 'maj7',
    'major9': 'maj7',
    'major11': 'maj7',
    'major13': 'maj7',
    'Δ7': 'maj7',
    'Δ9': 'maj7',
    'Δ11': 'maj7',
    'Δ13': 'maj7',
    'Δ': 'maj7',
    'j7': 'maj7',
    'j9': 'maj7',
    'j11': 'maj7',
    'j13': 'maj7',

    # Minor 7 chord qualities
    'min7': 'min7',
    'min9': 'min7',
    'min11': 'min7',
    'min13': 'min7',
    'm7': 'min7',
    'm9': 'min7',
    'm11': 'min7',
    'm13': 'min7',
    'minor7': 'min7',
    'minor9': 'min7',
    'minor11': 'min7',
    'minor13': 'min7',
    '-7': 'min7',
    '-9': 'min7',
    '-11': 'min7',
    '-13': 'min7',

    # Half-diminished 7 chord qualities
    'm7b5': 'hdim7',
    'min7b5': 'hdim7',
    'ø7': 'hdim7',
    'hdim7': 'hdim7',
    'half-diminished7': 'hdim7',

    # Diminished 7 chord qualities
    'dim7': 'dim7',
    'diminished7': 'dim7',
    'o7': 'dim7',

    # Augmented 7 chord qualities
    'aug7': 'aug7',
    'augmented7': 'aug7',
    '+7': 'aug7',

    # Major 6 chord qualities
    'maj6': 'maj6',
    'M6': 'maj6',
    'major6': 'maj6',
    'Δ6': 'maj6',
    '6': 'maj6',

    # Minor 6 chord qualities
    'min6': 'min6',
    'm6': 'min6',
    'minor6': 'min6',
    '-6': 'min6',

    # Sus 7 chord qualities
    'sus27': 'sus27',
    'sus47': 'sus47',
    'sus7': 'sus47',

    # Minor major 7 chord qualities
    'minmaj7': 'minmaj7',
    'mM7': 'minmaj7',
    'minorMajor7': 'minmaj7',
    '-maj7': 'minmaj7',
    '-Δ7': 'minmaj7',
    '-j7': 'minmaj7',
    
}

QUALITY_CLASSES = {
    'maj': 'MAJ',
    'maj6': 'MAJ',
    'min': 'MIN',
    'min6': 'MIN',
    'dim': 'DIM',
    'dim7': 'DIM7',
    'aug': 'AUG',
    'aug7': 'AUG',
    'sus2': 'SUS',
    'sus4': 'SUS',
    'sus27': 'SUS',
    'sus47': 'SUS',
    'sus7': 'SUS',
    'sus': 'SUS',
    '7': 'DOM',
    'maj7': 'MAJ7',
    'min7': 'MIN7',
    'hdim7': 'HDIM',
    'minmaj7': 'MINMAJ7',
}


ENHARMONIC_FLAT_EQUIVALENTS = {
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
    "A#": "Bb",
    "B#": "C",
    "E#": "F",
}

CHORD_SCALES = {
    # Major (Lydian)
    'MAJ': {0, 2, 4, 6, 7, 9, 11},
    'MAJ7': {0, 2, 4, 6, 7, 9, 11},
    # Minor (Dorian)
    'MIN': {0, 2, 3, 5, 7, 9, 10},
    'MIN7': {0, 2, 3, 5, 7, 9, 10},
    # Diminished (Whole-Half)
    'DIM': {0, 2, 3, 5, 6, 8, 9, 11},
    'DIM7': {0, 2, 3, 5, 6, 8, 9, 11},
    # Augmented (Whole Tone)
    'AUG': {0, 2, 4, 6, 8, 10},
    # Suspended (Mixolydian)
    'SUS': {0, 2, 5, 7, 9, 10},
    # Dominant 7 (Mixolydian)
    'DOM': {0, 2, 4, 5, 7, 9, 10},
    # Half-diminished 7 (Locrian #2)
    'HDIM': {0, 2, 3, 5, 6, 8, 10},
    # Minor major 7 (Melodic Minor)
    'MINMAJ7': {0, 2, 3, 5, 7, 9, 11},
}
import pytest
from app.utils.chords import Chord

@pytest.mark.parametrize(
    "label, root, quality, quality_class, extensions, bass",
    [
        # --- Basic triads ---
        ("C", "C", "maj", "MAJ", [], "C"),
        ("D-", "D", "min", "MIN", [], "D"),
        ("E+", "E", "aug", "AUG", [], "E"),
        ("Fo", "F", "dim", "DIM", [], "F"),

        # --- Sevenths ---
        ("G7", "G", "7", "DOM", [], "G"),
        ("Aj7", "A", "maj7", "MAJ7", [], "A"),
        ("B-7", "B", "min7", "MIN7", [], "B"),
        ("Co7", "C", "dim7", "DIM7", [], "C"),
        ("Dø7", "D", "hdim7", "HDIM", [], "D"),
        ("E-Δ7", "E", "minmaj7", "MINMAJ7", [], "E"),

        # --- 6th chords ---
        ("F6", "F", "maj6", "MAJ", [], "F"),
        ("G-6", "G", "min6", "MIN", [], "G"),

        # --- Suspended chords ---
        ("Asus2", "A", "sus2", "SUS", [], "A"),
        ("Bsus4", "B", "sus4", "SUS", [], "B"),
        ("Csus7", "C", "sus47", "SUS", [], "C"),

        # --- Extended chords (right-based accidentals) ---
        ("D79", "D", "7", "DOM", ["9"], "D"),
        ("E7911", "E", "7", "DOM", ["9", "11"], "E"),
        ("F7913", "F", "7", "DOM", ["9", "13"], "F"),
        ("G79b", "G", "7", "DOM", ["b9"], "G"),     # flat 9 suffix
        ("A7911#", "A", "7", "DOM", ["9","#11"], "A"),  # sharp 11 suffix
        ("Bb7913b", "Bb", "7", "DOM", ["9", "b13"], "Bb"),  # flat 13 suffix

        # --- Major and minor extended ---
        ("Cmaj79", "C", "maj7", "MAJ7", ["9"], "C"),
        ("Dmin7911#", "D", "min7", "MIN7", ["9","#11"], "D"),

        # --- Altered style (still suffix-based) ---
        ("E79b13b", "E", "7", "DOM", ["b9", "b13"], "E"),
        ("F79#13", "F", "7", "DOM", ["#9", "13"], "F"),
        ("G7911#13b", "G", "7", "DOM", ["9", "#11", "b13"], "G"),

        # --- Slash chords ---
        ("C7/E", "C", "7", "DOM", [], "E"),
        ("F-7/Ab", "F", "min7", "MIN7", [], "Ab"),
        ("Bbmaj79/D", "Bb", "maj7", "MAJ7", ["9"], "D"),

        # --- Edge cases: invalid roots / empty ---
        ("Xyz", "", "", "", [], ""),
        ("", "", "", "", [], ""),
        (None, "", "", "", [], ""),
    ]
)

def test_chord_parsing(label, root, quality, quality_class, extensions, bass):
    chord = Chord(label)

    if root == "":
        assert not chord.is_valid
    else:
        assert chord.is_valid is True
        assert chord.root == root
        assert chord.quality == quality
        assert chord.quality_class == quality_class
        assert chord.extensions == extensions
        assert chord.bass == bass
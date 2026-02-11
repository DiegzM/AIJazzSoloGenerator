// Utility for parsing chord symbols and converting them to MIDI note numbers
// Used for dynamically creating a chord track based on user-defined chords

const NOTE_TO_MIDI = {
  'C': 60, 'C#': 61, 'Db': 61,
  'D': 62, 'D#': 63, 'Eb': 63,
  'E': 64, 'F': 65, 'F#': 66, 'Gb': 66,
  'G': 67, 'G#': 68, 'Ab': 68,
  'A': 69, 'A#': 70, 'Bb': 70,
  'B': 71
};

const CHORD_INTERVALS = {
  'MAJ':    [0, 4, 7],
  'MAJ7':   [0, 4, 7, 11],
  'MIN':    [0, 3, 7],
  'MIN7':   [0, 3, 7, 10],
  'DOM':    [0, 4, 7, 10],
  'DIM':    [0, 3, 6],
  'DIM7':   [0, 3, 6, 9],
  'HDIM':   [0, 3, 6, 10],
  'AUG':    [0, 4, 8],
  'SUS':    [0, 5, 7],
};

// Convert MIDI number to note name (e.g., 60 -> C4)
const midiToNoteName = (midiNumber) => {
  const octave = Math.floor(midiNumber / 12) - 1;
  const noteIndex = midiNumber % 12;
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  return noteNames[noteIndex] + octave;
};

// Given a chord symbol, return an array of MIDI note numbers
export const getChordNotes = (root, quality_class) => {
    if (!root || !quality_class || root === 'NC') return [];
    const rootMidi = NOTE_TO_MIDI[root];
    const intervals = CHORD_INTERVALS[quality_class];
    if (!intervals) return [];
    return intervals.map(interval => midiToNoteName(rootMidi + interval));
}

// Generate accompaniment track based on chord progression
export const generateAccompanimentTrack = (beats, tempo = 120) => {
    const secondsPerBeat = 60 / tempo;
    const track = {
        name: "Chords",
        notes: []
    };
    
    beats.forEach(beat => {
        if (beat.root && beat.root !== 'NC') {  
            const chordNotes = getChordNotes(beat.root, beat.quality_class);
            chordNotes.forEach(note => {
                track.notes.push({
                    name: note,
                    time: (beat.bar * 4 + (beat.beat - 1)) * secondsPerBeat,
                    duration: secondsPerBeat,
                    velocity: 0.5
                });
            });
        }
    });

    return track;
};


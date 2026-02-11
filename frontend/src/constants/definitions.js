// GUI KEY OPTIONS → FINAL BACKEND KEY VALUES
export const KEY_OPTIONS = {
  "C Major": "C_MAJOR",
  "C Minor": "C_MINOR",
  "C# Major": "Db_MAJOR",
  "C# Minor": "Db_MINOR",
  "D♭ Major": "Db_MAJOR",
  "D♭ Minor": "Db_MINOR",
  "D Major": "D_MAJOR",
  "D Minor": "D_MINOR",
  "D# Major": "Eb_MAJOR",
  "D# Minor": "Eb_MINOR",
  "E♭ Major": "Eb_MAJOR",
  "E♭ Minor": "Eb_MINOR",
  "E Major": "E_MAJOR",
  "E Minor": "E_MINOR",
  "F Major": "F_MAJOR",
  "F Minor": "F_MINOR",
  "F# Major": "Gb_MAJOR",
  "F# Minor": "Gb_MINOR",
  "G♭ Major": "Gb_MAJOR",
  "G♭ Minor": "Gb_MINOR",
  "G Major": "G_MAJOR",
  "G Minor": "G_MINOR",
  "G# Major": "Ab_MAJOR",
  "G# Minor": "Ab_MINOR",
  "A♭ Major": "Ab_MAJOR",
  "A♭ Minor": "Ab_MINOR",
  "A Major": "A_MAJOR",
  "A Minor": "A_MINOR",
  "A# Major": "Bb_MAJOR",
  "A# Minor": "Bb_MINOR",
  "B♭ Major": "Bb_MAJOR",
  "B♭ Minor": "Bb_MINOR",
  "B Major": "B_MAJOR",
  "B Minor": "B_MINOR",
};

// GUI QUALITY → FINAL BACKEND QUALITY
export const QUALITY_MAP = {
  "maj": "MAJ",
  "min": "MIN",
  "°": "DIM",
  "°7": "DIM7",
  "+": "AUG",
  "sus": "SUS",
  "7": "DOM",
  "min7♭5": "HDIM",
  "min(maj7)": "MINMAJ7"
};

// GUI root names → canonical flat-only names
export const ROOT_MAP = {
  "C": "C",
  "C♯": "Db",
  "D♭": "Db",
  "D": "D",
  "D♯": "Eb",
  "E♭": "Eb",
  "E": "E",
  "F": "F",
  "F♯": "Gb",
  "G♭": "Gb",
  "G": "G",
  "G♯": "Ab",
  "A♭": "Ab",
  "A": "A",
  "A♯": "Bb",
  "B♭": "Bb",
  "B": "B"
};

// GUI root list
export const GUI_ROOTS = Object.keys(ROOT_MAP);

// GUI qualities — value is canonical backend quality_class
export const GUI_QUALITIES = [
  { label: "maj", value: "MAJ" },
  { label: 'maj7' , value: "MAJ7" },
  { label: "min", value: "MIN" },
  { label: "min7", value: "MIN7" },
  { label: "°", value: "DIM" },
  { label: "°7", value: "DIM7" },
  { label: "+", value: "AUG" },
  { label: "sus", value: "SUS" },
  { label: "7", value: "DOM" },
  { label: "min7♭5", value: "HDIM" },
  { label: "min(maj7)", value: "MINMAJ7" }
];

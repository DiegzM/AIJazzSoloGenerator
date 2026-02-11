import { useState, useRef, useEffect } from "react";
import { Midi } from '@tonejs/midi';
import * as Soundfont from 'soundfont-player';
import { generateAccompanimentTrack } from "../../utils/chordParser";
import MidiProgressBar from "./MidiProgressBar";

// Component to playback midiData and download MIDI file
export default function MidiPlayer({ midiData, beats, tempo = 120, fileName = "Jazz_Solo" }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [chordsEnabled, setChordsEnabled] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalTime, setTotalTime] = useState(0);

  // Store MIDI context and instrument

  const soloInstrumentRef = useRef(null);
  const chordInstrumentRef = useRef(null);
  const audioContextRef = useRef(null);

  // Refs for tracking playback state
  const midiDataRef = useRef(null);
  const chordTrackRef = useRef(null);
  const durationRef = useRef(0);
  const startTimeRef = useRef(0);
  const progressIntervalRef = useRef(null);

  // Separate gain nodes for solo and chords
  const soloGainRef = useRef(null);
  const chordGainRef = useRef(null);

  // Refs for scheduled events
  const scheduledNotesRef = useRef({ solo: [], chords: [] });
  const stopTimeoutRef = useRef(null);

  // Get audio context
  const getAudioContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      
      // Create gain nodes
      soloGainRef.current = audioContextRef.current.createGain();
      soloGainRef.current.connect(audioContextRef.current.destination);

      chordGainRef.current = audioContextRef.current.createGain();
      chordGainRef.current.connect(audioContextRef.current.destination);
    }
    return audioContextRef.current;
  };

  // Toggle chord volume
  useEffect(() => {
    if (chordGainRef.current) {
      chordGainRef.current.gain.value = chordsEnabled ? 1 : 0;
    }
  }, [chordsEnabled]);

  // Load instruments
  const loadInstruments = async () => {
    const ac = getAudioContext();

    const [solo, chords] = await Promise.all([
      soloInstrumentRef.current || Soundfont.instrument(ac, 'alto_sax', { destination: soloGainRef.current }),
      chordInstrumentRef.current || Soundfont.instrument(ac, 'acoustic_grand_piano', { destination: chordGainRef.current })
    ]);

    soloInstrumentRef.current = solo;
    chordInstrumentRef.current = chords;
  };


  // Schedule notes from a specific time offset (for seeking)
  const scheduleNotesFromOffset = (offsetSeconds) => {
    const now = getAudioContext().currentTime;

    // Stop any currently playing notes
    midiDataRef.current.tracks.forEach(track => {
      track.notes.forEach(note => {
        if (note.time >= offsetSeconds) {
          const scheduled = soloInstrumentRef.current.play(
            note.name,
            now + (note.time - offsetSeconds),
            { duration: note.duration, gain: note.velocity }
          );
          scheduledNotesRef.current.solo.push(scheduled);
        }
      });
    });

    if (chordTrackRef.current) {
      chordTrackRef.current.notes.forEach(note => {
        if (note.time >= offsetSeconds) {
          const scheduled = chordInstrumentRef.current.play(
            note.name,
            now + (note.time - offsetSeconds),
            { duration: note.duration, gain: note.velocity }
          );
          scheduledNotesRef.current.chords.push(scheduled);
        }
      });
    }
  };

  // Start progress bar updates
  const startProgressUpdates = () => {
    if (progressIntervalRef.current) clearInterval(progressIntervalRef.current);
    progressIntervalRef.current = setInterval(() => {
      const elapsed = getAudioContext().currentTime - startTimeRef.current;
      const percentage = Math.min((elapsed / durationRef.current) * 100, 100);
      setProgress(percentage);
      setCurrentTime(Math.min(elapsed, durationRef.current));

      if (percentage >= 100) {
        stopMidi();
      }
    }, 50); // Update every 50ms
  };

  // Handle seeking
  const handleSeek = (percentage) => {
    setProgress(percentage);

    // Update currenttime
    const seekTime = (percentage / 100) * durationRef.current;
    setCurrentTime(seekTime);

    if (!isPlaying || !midiDataRef.current) return;

    // Stop current notes
    scheduledNotesRef.current.solo.forEach(note => note.stop());
    scheduledNotesRef.current.chords.forEach(note => note.stop());
    scheduledNotesRef.current = { solo: [], chords: [] };
    soloInstrumentRef.current?.stop();
    chordInstrumentRef.current?.stop();

    if (stopTimeoutRef.current) {
      clearTimeout(stopTimeoutRef.current);
    }

    // Calculate new offset
    const offsetSeconds = (percentage / 100) * durationRef.current;
    startTimeRef.current = getAudioContext().currentTime - offsetSeconds;

    // Reschedule notes from new offset
    scheduleNotesFromOffset(offsetSeconds);

    const remainingDuration = durationRef.current - offsetSeconds;
    stopTimeoutRef.current = setTimeout(() => {
      stopMidi();
    }, remainingDuration * 1000);
  }

  // Play MIDI
  const playMidi = async () => {
    setIsLoading(true);

    try {
      await loadInstruments();
      await getAudioContext().resume();
      
      if (!midiDataRef.current) {
        const response = await fetch(midiData);
        const arrayBuffer = await response.arrayBuffer();
        const midi = new Midi(arrayBuffer);
        midiDataRef.current = midi;
      } 

      if (beats?.length && !chordTrackRef.current) {
        chordTrackRef.current = generateAccompanimentTrack(beats, tempo);
      }

      durationRef.current = Math.max(
        midiDataRef.current.duration,
        (beats?.length || 0) * (60 / tempo) * 4 // Estimate chord track duration
      );

      // Set total time for progress bar
      setTotalTime(durationRef.current);

      const startOffset = (progress / 100) * durationRef.current;
      startTimeRef.current = getAudioContext().currentTime - startOffset;

      scheduleNotesFromOffset(startOffset);
      startProgressUpdates();

      const remaining = durationRef.current - startOffset;
      stopTimeoutRef.current = setTimeout(() => {
        stopMidi();
      }, remaining * 1000);
    }
    catch (error) {
      console.error("Error during MIDI playback:", error);
      setIsPlaying(false);
    }
    finally {
      setIsLoading(false);
    }
  };

  // Stop MIDI playback
  const stopMidi = () => {
    // Clear auto-stop timeout
    if (stopTimeoutRef.current) {
      clearTimeout(stopTimeoutRef.current);
    }

    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }

    // Stop all scheduled notes
    scheduledNotesRef.current.solo.forEach(note => note.stop());
    scheduledNotesRef.current.chords.forEach(note => note.stop());
    scheduledNotesRef.current = { solo: [], chords: [] };

    soloInstrumentRef.current?.stop();
    chordInstrumentRef.current?.stop();

    setIsPlaying(false);
  }
    
  
  const handlePlayPause = async () => {
    if (isPlaying) {
      stopMidi();
    } else {
      setIsPlaying(true);
      await playMidi();
    }
  };

  // Load midi data on mount or when midiData changes to get duration and prepare for playback
  useEffect(() => {
    const loadMidiData = async () => {
      if (!midiData) return;

      setIsLoading(true);
      try {
        const response = await fetch(midiData);
        const arrayBuffer = await response.arrayBuffer();
        const midi = new Midi(arrayBuffer);
        midiDataRef.current = midi;

        const duration = Math.max(
          midi.duration,
          (beats?.length || 0) * (60 / tempo) * 4 // Estimate chord track duration
        );
        durationRef.current = duration;
        // Set total time for progress bar
        setTotalTime(duration);
      } catch (error) {
        console.error("Error loading MIDI data:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadMidiData();
  }, [midiData, beats, tempo]);

  // Clear cached data when midiData changes
  useEffect(() => {
    midiDataRef.current = null;
    chordTrackRef.current = null;
    setProgress(0);
  }, [midiData]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if(progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
      if (stopTimeoutRef.current) {
        clearTimeout(stopTimeoutRef.current);
      }
      stopMidi();
    };
  }, []);

  // Format time in mm:ss
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="w-full mb-8 mt-8 p-6 bg-transparent shadow-md rounded-xl border border-gray-800 text-white flex flex-col items-center">
      <h2 className="text-3xl p-2 font-semibold mb-4">MIDI PLAYER</h2>

      {/* Download MIDI Button */}
      <a
          href={midiData}
          download={`${fileName}.mid`}
          className="mb-4 px-4 py-2 bg-amber-500 text-gray-900 font-semibold rounded hover:bg-amber-600 transition"
      >
          Download MIDI File
      </a>

      {/* Toggle play chords */}
      <div className="mb-4">
        <label className="inline-flex items-center">
          <input
            type="checkbox"
            checked={chordsEnabled}
            onChange={() => setChordsEnabled(!chordsEnabled)}
            className="form-checkbox h-5 w-5 text-amber-500"
          />
          <span className="ml-2 text-gray-300">Play Chord Accompaniment</span>
        </label>
      </div>

      {/* Progress Bar */}
      <MidiProgressBar progress={progress} onSeek={handleSeek} />

      {/* Time Display  (bar width) */}
      <div className="mb-4 text-gray-300">
        {formatTime(currentTime)} / {formatTime(totalTime)}
      </div>

      {/* Play/Pause Button */}
      <button
        onClick={handlePlayPause}
        className={`
          px-6 py-2 rounded-xl font-semibold text-lg transition-all duration-200
          ${isPlaying 
            ? "bg-red-600 hover:bg-red-500 text-white" 
            : "bg-amber-500 hover:bg-amber-400 text-gray-900"
          }
        `}
      >
        {isPlaying ? "Pause Playback" : "Play MIDI"}
      </button>
    </div>
  );
}   
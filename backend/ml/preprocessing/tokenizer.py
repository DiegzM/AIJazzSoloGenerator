
from rich.console import Console
import json

# Tokenizer for encoding solos into token ids
class Tokenizer:

    # Constructor (initialize vocabulary)
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = {
                "<PAD>": 0,
                "<SOS>": 1,
                "<EOS>": 2,
            }
        else:
            self.vocab = vocab

    # Build vocabulary from sequences of tokens
    def build_vocab_from_sequences(self, sequences):
        for encoder_input, decoder_target in sequences:
            for token in encoder_input:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
            for token in decoder_target:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    # Save vocabulary to file
    def save_vocab(self, filepath):
        with open(filepath, 'w') as f:
            for token, token_id in self.vocab.items():
                f.write(f"{token}\t{token_id}\n")

    # Load vocabulary from file
    def load_vocab(self, filepath):
        vocab = {}
        with open(filepath, "r") as f:
            for line in f:
                token, token_id = line.strip().split("\t")
                vocab[token] = int(token_id)
        self.vocab = vocab

    # Encode solo into token sequence pairs (encoder (meta + chords) and decoder (notes))
    def encode_solo(self, solo, beats, solo_info):

        # Start of phrase
        solo = json.loads(solo.to_json(orient="records"))
        beats = json.loads(beats.to_json(orient="records"))
        solo_info = json.loads(solo_info.iloc[0].to_json())
        
        encoder_tokens = self.encode_chord_timeline(beats, solo_info)
        decoder_tokens = self._encode_melody(solo, beats)

        if len(encoder_tokens) > 700 or len(decoder_tokens) > 4000:
            print("Skipping melid", solo_info['melid'], "due to length:", len(encoder_tokens), len(decoder_tokens))
            return None
        else:
            return {
                "encoder_input": encoder_tokens,
                "decoder_target": decoder_tokens,
            }

    # Encode metadata and chord timeline into tokens
    def encode_chord_timeline(self, beats, solo_info):

        tokens = ['<SOS>']

        # Encode key from solo_info
        key = solo_info['key']
        tokens.append(f'KEY_{key}')

        # Encode style from solo_info
        style = solo_info['style']
        tokens.append(f'STYLE_{style}')

        # Encode tempo (int)
        tempo = int(round(solo_info['avgtempo'] / 10) * 10)
        tokens.append(f'TEMPO_{tempo}')

        # Encode chords
        current_bar = -1
        current_beat = 0
        current_chord = "NC"

        for row in beats:
            bar = int(row['bar'])
            beat = int(row['beat'])
            chord = row['root'] + "_" + row['quality_class']  if row['root'] else "NC"

            # If bar changes, add BAR token
            if bar != current_bar:
                tokens.append(f'BAR')
                current_bar = bar
                current_beat = 0  # Reset beat on new bar
            
            # If theres a chord, add its beat token and chord token
            if chord != current_chord:
                tokens.append(f'BEAT_{beat}')
                tokens.append(f'CHORD_{chord}')
                current_chord = chord

        tokens.append('<EOS>')

        return tokens

    def _encode_melody(self, solo, beats):
        tokens = ["<SOS>"]
        
        current_onset = 0.0
        current_bar = 0
        
        for row in solo:
            onset = row['onset']
            duration = row['duration']
            pitch = int(row['pitch'])
            
            # Check if we've crossed into new bar(s)
            note_bar = int(onset // 4)  # Assuming 4 beats per bar
            while current_bar < note_bar:
                # Add TIME_SHIFT to reach end of current bar if needed
                bar_end = (current_bar + 1) * 4.0
                if bar_end > current_onset:
                    time_to_bar_end = bar_end - current_onset
                    while time_to_bar_end > 0:
                        shift = min(time_to_bar_end, 2.0)
                        tokens.append(f'TIME_SHIFT_{int(shift * 24)}')
                        time_to_bar_end -= shift
                    current_onset = bar_end
                
                tokens.append('BAR')
                current_bar += 1
            
            # Add TIME_SHIFT tokens if onset has advanced within the bar
            if onset > current_onset:
                time_shift_amount = onset - current_onset
                while time_shift_amount > 0:
                    shift = min(time_shift_amount, 2.0)
                    tokens.append(f'TIME_SHIFT_{int(shift * 24)}')
                    time_shift_amount -= shift
                current_onset = onset
            
            # Add NOTE and DURATION tokens
            tokens.append(f'NOTE_{pitch}')
            tokens.append(f'DURATION_{int(duration * 24)}')
        
        # Add remaining bars if any
        last_bar_in_beats = int(beats[-1]['bar'])
        while current_bar < last_bar_in_beats:
            bar_end = (current_bar + 1) * 4.0
            if bar_end > current_onset:
                time_to_bar_end = bar_end - current_onset
                while time_to_bar_end > 0:
                    shift = min(time_to_bar_end, 2.0)
                    tokens.append(f'TIME_SHIFT_{int(shift * 24)}')
                    time_to_bar_end -= shift
                current_onset = bar_end
            tokens.append('BAR')
            current_bar += 1
        
        tokens.append("<EOS>")
        return tokens
    
    # Encode chords only for inference (Beats and solo info are just basic dictionaries)
    def encode_chords_only(self, beats, solo_info):
        tokens = ['<SOS>']

        # Encode key from solo_info
        key = solo_info['key']
        tokens.append(f'KEY_{key}')

        # Encode style from solo_info
        style = solo_info['style']
        tokens.append(f'STYLE_{style}')

        # Encode tempo (int)
        tempo = int(round(solo_info['avgtempo'] / 10) * 10)
        tokens.append(f'TEMPO_{tempo}')

        # Encode chords
        current_bar = -1
        current_beat = 0
        current_chord = "NC"
        for row in beats:
            bar = int(row['bar'])
            beat = int(row['beat'])
            chord = row['root'] + "_" + row['quality_class']  if row['root'] else "NC"

            # If bar changes, add BAR token
            if bar != current_bar:
                tokens.append(f'BAR')
                current_bar = bar
                current_beat = 0  # Reset beat on new bar
            
            # If theres a chord, add its beat token and chord token
            if chord != current_chord:
                tokens.append(f'BEAT_{beat}')
                tokens.append(f'CHORD_{chord}')
                current_chord = chord

        tokens.append('<EOS>')

        return tokens
    
    # Decode melody to midi note sequence {onset, pitch, duration}
    def decode_melody(self, melody_tokens):
        events = []

        onset = 0.0
        pitch = None
        duration = 0.0
        
        for token in melody_tokens:
            if token.startswith("NOTE_"):
                pitch = int(token.split("_")[-1])
            elif token.startswith("DURATION_"):
                duration = int(token.split("_")[-1]) / 24.0

                if pitch is not None and duration > 0.0:
                    events.append({
                        "onset": onset,
                        "pitch": pitch,
                        "duration": duration
                    })
                    pitch = None
                    duration = 0.0
            elif token.startswith("TIME_SHIFT_"):
                onset += int(token.split("_")[-1]) / 24.0
        return events
            
        

    def id_to_token(self, idx):
        for token, token_id in self.vocab.items():
            if token_id == idx:
                return token
        return None
    
    def token_to_id(self, token):
        return self.vocab.get(token, None)
    
    def tokens_to_ids(self, tokens):
        return [self.token_to_id(token) for token in tokens]
    
    def encode_to_ids(self, encoder_tokens, decoder_tokens):
        return (
            [self.token_to_id(t) for t in encoder_tokens],
            [self.token_to_id(t) for t in decoder_tokens]
        )

import os
import typing

import numpy as np
import pedalboard
import soundfile as sf
from music21 import stream, note, metadata, tempo, tie, instrument, clef
import logging
import common

common.logger.configure_logger()

MUSESCORE_PATH = r"MuseScore\MuseScorePortable\MuseScorePortable.exe"

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

major_intervals = [2, 2, 1, 2, 2, 2, 1]

MAJOR_SCALES = {}

for i, root in enumerate(NOTE_NAMES):
    scale = [root]
    idx = i
    for step in major_intervals[:-1]:
        idx = (idx + step) % 12
        scale.append(NOTE_NAMES[idx])
    MAJOR_SCALES[root] = scale

NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}


def get_subclasses(parent_class) -> list:
    subclasses = set()
    work = [parent_class]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)

    return list(subclasses)


instruments = get_subclasses(instrument.Instrument)
instrument_names = [i.__name__.lower() for i in instruments]
instrument_display_names = [i.__name__ for i in instruments]

clefs = get_subclasses(clef.Clef)
clefs_names = [i.__name__.lower() for i in clefs]
clefs_display_names = [i.__name__ for i in clefs]


def analyze_signal(x, sr, segment_duration, n_freqs):
    seg_len = int(segment_duration * sr)
    n_segments = len(x) // seg_len

    frequencies = np.zeros((n_segments, n_freqs))
    amplitudes = np.zeros(n_segments)

    freqs = np.fft.rfftfreq(seg_len, d=1/sr)

    for i in range(n_segments):
        seg = x[i * seg_len:(i + 1) * seg_len]

        max_amp = np.max(np.abs(seg))

        window = np.hanning(seg_len)
        seg_win = seg * window

        spectrum = np.fft.rfft(seg_win)
        mag = np.abs(spectrum)

        idx = np.argpartition(mag, -n_freqs)[-n_freqs:]
        idx = idx[np.argsort(-mag[idx])]  # sort strongest first

        prominent_freqs = freqs[idx]

        frequencies[i] = prominent_freqs
        amplitudes[i] = max_amp

    return frequencies, amplitudes


def freq_to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)


def midi_to_note_name(midi_num):
    note = NOTE_NAMES[int(midi_num) % 12]
    octave = int(midi_num // 12) - 1
    return note, octave


def note_to_frequency(note_octave: str) -> float:
    octave = int(note_octave[-1])
    name = note_octave[:-1]

    n = NOTE_NAMES.index(name)
    semitones = n - NOTE_NAMES.index('A') + 12 * (octave - 4)
    return 440 * (2 ** (semitones / 12))


def get_note_duration(bpm, note_fraction):
    beats_per_whole_note = 4
    beats = beats_per_whole_note * note_fraction

    duration_seconds = (60 / bpm) * beats
    return duration_seconds


def is_the_same(current, part):
    try:
        last = part[-1]
    except IndexError:
        return False

    if isinstance(current, note.Rest) and isinstance(last, note.Rest):
        return True

    if isinstance(current, note.Note) and isinstance(last, note.Note):
        return current.name == last.name and current.octave == last.octave

    return False


def signal_to_notes(frequencies: np.ndarray, part, amplitudes, min_frequency, max_frequency, precision, set_velocity: bool = True, min_volume=0.5):
    previous_note = None
    for frequency, amplitude in zip(frequencies, amplitudes):
        if min_frequency < frequency < max_frequency:
            midi_val = freq_to_midi(frequency)
            name, octave = midi_to_note_name(round(midi_val))
            prospective_note = note.Note("{}{}".format(name, int(octave)), quarterLength=precision * 4)

            if set_velocity:
                if min_volume != 0:
                    amplitude = min_volume + amplitude * (1 - min_volume)
                velocity = int(np.clip(amplitude * 127, 1, 127))
                prospective_note.volume.velocity = velocity

            previous_note = prospective_note

        else:
            if previous_note is None:
                prospective_note = note.Rest(quarterLength=precision * 4)
            else:
                prospective_note = note.Note(previous_note.pitch, quarterLength=precision * 4)
                if previous_note.tie is None:
                    previous_note.tie = tie.Tie('start')
                prospective_note.tie = tie.Tie('continue')

        if is_the_same(prospective_note, part):
            part[-1].quarterLength += prospective_note.quarterLength

        else:
            part.append(prospective_note)


def estimate_key(score) -> typing.Tuple[str, float]:
    counters = {name: 0 for name in NOTE_NAMES}
    total_notes = 0

    for part in score:
        if isinstance(part, stream.Part):
            if isinstance(part.getInstrument(), instrument.Percussion) and not isinstance(part.getInstrument(), instrument.PitchedPercussion):
                continue

            for n in part:
                if isinstance(n, note.Note):
                    for counter in counters:
                        counters[counter] += n.name in MAJOR_SCALES[counter]

                    total_notes += 1

    max_key, max_count = max(counters.items(), key=lambda x: x[1])

    return max_key, max_count / total_notes


def remap_song(score, key: str):
    valid_notes = MAJOR_SCALES[key]

    for part in score:
        if isinstance(part, stream.Part):
            if isinstance(part.getInstrument(), instrument.Percussion) and not isinstance(part.getInstrument(), instrument.PitchedPercussion):
                continue

            for i, n in enumerate(part):
                if isinstance(n, note.Note):
                    if not n.name in valid_notes:
                        n.name = closest_note(n.name, valid_notes)


def closest_note(note, valid_notes):
    target_val = NOTE_TO_SEMITONE[note]
    valid_vals = [NOTE_TO_SEMITONE[n] for n in valid_notes]

    distances = [min(abs(target_val - v), 12 - abs(target_val - v)) for v in valid_vals]

    min_idx = distances.index(min(distances))

    return valid_notes[min_idx]


def get_average_octave(part):
    octaves = []
    for i, n in enumerate(part):
        if isinstance(n, note.Note):
            if n.octave is not None:
                octaves.append(n.octave)

    return np.median(octaves)


def shift_notes(part, delta):
    for i, n in enumerate(part):
        if isinstance(n, note.Note):
            if n.octave is not None:
                n.octave += delta


class Track:
    def __init__(
            self,
            render_instrument: instrument.Instrument,
            shift_clef: clef.PitchClef = None,
            median_octave: typing.Union[int, None] = 'auto'
    ):
        self.instrument = render_instrument
        self.clef = shift_clef
        self.median_octave = median_octave

        if self.clef is None:
            if isinstance(self.instrument, instrument.ElectricBass):
                self.clef = clef.BassClef

        if self.median_octave == "auto":
            if isinstance(self.clef, clef.BassClef):
                self.median_octave = 2

            elif isinstance(self.clef, clef.TrebleClef):
                self.median_octave = 4

            else:
                self.median_octave = None


class SoundToMusicConfig(common.SavableObject):
    def __init__(
            self,
            tempo: float,
            precision: float,
            tracks: typing.List[Track],
            input_path: str,
            output_directory: str,
            effects_chain: pedalboard.Pedalboard = None,
            min_frequency: typing.Union[str, float] = "A1",
            max_frequency: typing.Union[str, float] = "G5",
            musescore_path: str = None,
            note_velocity: bool = False
    ):
        self.tempo = tempo
        self.precision = precision
        self.tracks = tracks
        self.effects_chain = effects_chain
        self.input_path = input_path
        self.output_directory = output_directory
        self.min_frequency: float = min_frequency
        self.max_frequency: float = max_frequency
        self.musescore_path = MUSESCORE_PATH if musescore_path is None else musescore_path
        self.note_velocity = note_velocity

        if self.effects_chain is None:
            self.effects_chain = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.25)])

        if isinstance(self.min_frequency, str):
            self.min_frequency = note_to_frequency(self.min_frequency)
        if isinstance(self.max_frequency, str):
            self.max_frequency = note_to_frequency(self.max_frequency)

    @classmethod
    def from_argv(cls):
        parser = common.argv.get_argparse(
            "Converts an audio clip to several concurrent tracks of music notes, "
            "and is then rendered with MuseScore (if installed). "
            "An optional effects chain can be applied to the result if desired."
        )
        parser.add_argument("result", help="The name of the result to render as music")
        parser.add_argument("tempo", type=float, help="The tempo to use when converting the sound to music")
        parser.add_argument("--precision", type=float, default=1/8, help="The minimum note duration that can be used")
        parser.add_argument("tracks", nargs='+', help="A list of instruments to use for each track")
        parser.add_argument(
            "--musescore",
            help="The path to MuseScore. See the README for setup instructions. "
                 "If omitted, only a .midi and MusicXML file will be produced",
            default=MUSESCORE_PATH
        )

        parser.add_argument(
            '--standard_velocity', action='store_true',
            help="If enabled, all notes will assume a constant velocity instead of being inferred by the amplitude"
        )

        args = parser.parse_args()

        tracks = []
        for track in args.tracks:
            if "|" in track:
                split_track = track.split("|")
                selected_instrument = split_track[0]
                selected_clef = split_track[1]

            else:
                selected_instrument = track
                selected_clef = clef.TrebleClef.__name__

            if selected_instrument.lower() not in instrument_names:
                raise ValueError("Requested instrument '{}' is not supported. Please pick from the following list: {}".format(
                    selected_instrument, ", ".join(instrument_display_names)
                ))

            index = instrument_names.index(selected_instrument.lower())
            selected_instrument = instruments[index]()

            if selected_clef.lower() not in clefs_names:
                raise ValueError("Requested clef '{}' is not supported. Please pick from the following list: {}".format(
                    selected_clef, ", ".join(clefs_display_names)
                ))
            index = clefs_names.index(selected_clef.lower())
            selected_clef = clefs[index]()

            tracks.append(Track(selected_instrument, selected_clef))

        return cls(
            args.tempo, args.precision, tracks,
            os.path.join("output", args.result, "raw_audio.wav"), os.path.join("output", args.result),
            musescore_path=args.musescore, note_velocity=not args.standard_velocity
        )


def sound_to_music(config: SoundToMusicConfig):
    logging.info("Reading input audio file '{}'...".format(os.path.abspath(config.input_path)))
    x, sr = sf.read(config.input_path)
    if x.ndim > 1:
        x = x.mean(axis=1)

    note_duration = get_note_duration(config.tempo, config.precision)

    logging.info("Extracting frequencies for bins of length {} ({} notes at {}bpm) into {} tracks...".format(
        note_duration, config.precision, config.tempo, len(config.tracks)
    ))
    freq, amp = analyze_signal(x, sr, note_duration, n_freqs=len(config.tracks))

    logging.info("Initializing score metadata...")
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = os.path.splitext(os.path.basename(config.input_path))[0]
    score.metadata.composer = "Audiovisual Abstraction"

    metronome = tempo.MetronomeMark(number=config.tempo)

    logging.info("Creating each instrument...")
    for i, track in enumerate(config.tracks):
        if track.instrument is None:
            continue

        logging.info(
            "Instrument #{} - {} in {}".format(
                i + 1, track.instrument.__class__.__name__, track.clef.__class__.__name__
            )
        )

        part = stream.Part()
        part.append(track.instrument)
        part.id = "P{}".format(i)
        part.append(metronome)

        logging.info("Discretizing frequencies as music notes")
        signal_to_notes(freq[:, i], part, amp, config.min_frequency, config.max_frequency, config.precision, set_velocity=config.note_velocity)

        if track.clef is not None:
            logging.info("Overriding the track's clef to be {}".format(track.clef.__class__.__name__))
            part.clef = track.clef

        if track.median_octave is not None:
            logging.info("Shifting the octave to have a median of {}".format(track.median_octave))
            shift_notes(part, int(track.median_octave - get_average_octave(part)))

        logging.info("Adding the track to the score")
        score.insert(0, part)

    best_key, best_key_score = estimate_key(score)
    logging.info("Generated notes are {:.2f}% within the {} Major scale".format(best_key_score * 100, best_key))

    logging.info("Shifting notes to be fully within the {} Major scale".format(best_key))
    remap_song(score, best_key)

    logging.info("Saving track as MusicXML...")
    score.write("musicxml", os.path.join(config.output_directory, "music_audio.xml"))

    logging.info("Saving track as MIDI...")
    score.write("midi", os.path.join(config.output_directory, "music_audio.midi"))

    logging.info("Rendering audio with MuseScore...")
    os.system('{} "{}" -o "{}"'.format(
        config.musescore_path,
        os.path.join(config.output_directory, "music_audio.xml"),
        os.path.join(config.output_directory, "music_audio.wav")
    ))

    logging.info("Applying effects pedalboard to result...")
    with pedalboard.io.AudioFile(os.path.join(config.output_directory, "music_audio.wav")) as f:
        with pedalboard.io.AudioFile(os.path.join(config.output_directory, "music_audio_effects.wav"), 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                o.write(config.effects_chain(chunk, f.samplerate, reset=False))


if __name__ == '__main__':
    cfg = SoundToMusicConfig.from_argv()

    sound_to_music(cfg)

# Audiovisual Abstraction Through Element Rearrangement
The official code repository for the paper "Audiovisual Abstraction Through Element Rearrangement" by Christopher Palazzolo, Oliver van Kaick, and David Mould.

Presented at [ARTeFACTo 2026](https://2026.artefacto-conf.org/).

[Project Page (WIP)] 
[[Summary Video](https://youtu.be/JrfEbTIg5kc)] 
[[Long Video](https://youtu.be/Hmu3my3gXU8)]

## Installation
First, clone the repository to your local machine.

```shell
git clone https://github.com/ChrisPGraphics/Audiovisual-Abstraction-Through-Element-Rearrangement
cd Audiovisual-Abstraction-Through-Element-Rearrangement
```

Assuming you have Python installed (we use Python 3.11.5), follow these instructions to prepare your local machine to use our code. 

```shell
pip install -r requirements.txt
```

If you are using a Linux-based OS, you may need to also run the following commands to install dependencies for Open-CV2.

```shell
apt-get update
apt-get install ffmpeg libsm6 libxext6 -y
```

### MuseScore to Render Music (Optional)
If you want to render the music notes to a sound file, you will need to install software that can do so.
Note that this is only required for post-processing generated music notes to audio (`sound_to_music.py`).
We suggest [MuseScore](https://musescore.org/en) because it is free, and it can be automatically integrated in our pipeline.
Other options like [GuitarPro](https://www.guitar-pro.com/) will work too, but can not be integrated automatically.

You can download MuseScore from https://musescore.org/en/download (we use "MuseScore Portable EXE". Other types have not been tested).
Any download version works, but the default download button will install MuseHub which is not required. 
We used "MuseScore Portable EXE", but any option should still work.

If you download "MuseScore Portable", run the installer to get the MuseScorePortable executable. It should be saved with the following directory structure:
```
Audiovisual-Abstraction-Through-Element-Rearrangement/
  + MuseScore/
  |  + MuseScorePortable/
  |  |  + App/
  |  |  |  + ...
  |  |  + Data/
  |  |  |  + ...
  |  |  + Other/
  |  |  |  + ...
  |  |  + help.html
  |  |  + MuseScorePortable.exe
```

By default, the MuseScore path is set to:
```python
MUSESCORE_PATH = r"MuseScore\MuseScorePortable\MuseScorePortable.exe"
```

The default can be changed in line 13 of `sound_to_music.py`, or by using the `--musescore` command line argument of `sound_to_music.py`.

## AV Abstraction
This section describes how to use the pipeline on one or more video sources, render the audio, and merge the result to get a final video.

### Main Pipeline
To run the main pipeline, use the following command

```shell
python render_av.py [exemplar] [output_name] [duration]
```

By default, `exemplar` is the path to the single video source. To use many video sources, the following flags can be used:
- `--flow_video` - The video to extract optical flow from the defines how the polygons move
- `--color_video` - The video where the polygon colors are sampled from
- `--background_color_video` - The video where the background colors are sampled from
- `--audio` - The sound file that will be used. Omit to use the soundtrack of `exemplar`. If the path to a video is provided, the audio will be automatically extracted.

`exemplar` is used for any unspecified video, and is also the video the polygon shapes come from.

This will create a directory in `Audiovisual-Abstraction-Through-Element-Rearrangement/output/[output_name]` containing `raw_video.avi` and `raw_audio.wav`.
`raw_video.avi` is ready to go for the final merge. `raw_audio.wav` needs to be post-processed which is described in the next section (you can also use it as is, but it generally does not sound very good on its own).

You can use the following to display the help text to list all options

```shell 
python render_av.py -h
```

### Post-Processing the Sound
This section describes how to render `raw_audio.wav` in an abstract way. There are two choices for this, render as [Music Notes](#rendering-as-music-notes) or with an [Effects Chain](#rendering-with-effects-chain).

#### Rendering as Music Notes
To render `raw_audio.wav` as music notes, use the following command

```shell
python sound_to_music.py [result] [tempo] [tracks]
```

`result` is the `output_name` used when you called `render_av.py`. 
`tempo` is the tempo to use in the result.

`tracks` is a space-seperated list of the instruments to be used to render each track. 
The first track comes from the most prominent frequency in the soundtrack. Optionally, after the track name, you can put a `|` and specify the clef to shift the notes to.
An example value would be `ElectricGuitar ElectricBass|BassClef Piano`. 
`ElectricGuitar` is created from the most prominent frequencies. 
`ElectricBass|BassClef` is an electric bass guitar shifted to the bass clef. 
The last track is a piano. 

This will create `music_audio.wav` in the same directory as `raw_audio.wav` and `raw_video.avi` (if [MuseScore](#musescore-to-render-music-optional) is configured).
Regardless of MuseScore, it will create `music_audio.xml` containing the sheet music XML program that can be opened and manually rendered in other programs, and `music_audio.midi` which is a midi approximation of the final result.
Note that the midi file can not be merged directly. You would need to convert to a sound file before merging.
Finally, `music_audio_effects.wav` will be created which is `music_audio.wav` but with some light reverb.


You can use the following to display the help text to list all options.

```shell 
python sound_to_music.py -h
```

#### Rendering with Effects Chain
To apply an effects chain to `raw_audio.wav`, use the following command

```shell
python sound_to_effects.py [result]
```
`result` is the `output_name` used when you called `render_av.py`. 

This will create `effects_audio.wav` in the same directory as `raw_audio.wav` and `raw_video.avi` 

The effects chain must be set in `sound_to_effects.py` (In the `SoundToEffectsConfig` class under `def from_argv(cls)`, around line 28). There are no other arguments for this script.

### Merging the Final Audio and Video
Once both the audio and video are ready, you can use the following command to merge them

```shell
python merge_av.py [result] [sound_file]
```
`result` is the `output_name` used when you called `render_av.py`. 
`sound_file` is the name of the sound file to merge. 
If you used an effects chain, this will be `effects_audio.wav`. 
If you used the sound to music, then this will be either `music_audio.wav` or `music_audio_effects.wav` (If [MuseScore](#musescore-to-render-music-optional) is configured).

By default, this will create `result.mp4` in the `output` directory (`Audiovisual-Abstraction-Through-Element-Rearrangement/output/[result]/result.mp4`)
The name can be overridden with the `--name NAME` command line option. The file extension dictates the resulting video format.

## Citation
```
Coming Soon
```

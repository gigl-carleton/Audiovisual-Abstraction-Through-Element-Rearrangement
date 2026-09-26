import datetime
import logging
import os
import typing

import moviepy


def split_audio_video(filename: str, overwrite_audio: bool = False) -> typing.Tuple[str, str]:
    audio_path = os.path.splitext(filename)[0] + ".wav"

    if not overwrite_audio and os.path.isfile(audio_path):
        logging.warning("Audio file '{}' exists! Using this instead of extracting audio from video!".format(
            os.path.abspath(audio_path)
        ))

    else:
        logging.info("Reading video file '{}' into memory".format(os.path.abspath(filename)))
        clip = moviepy.VideoFileClip(filename)

        if clip.audio is None:
            logging.warning("Selected video file does not have sound! Skipping audio synthesis step.")
            clip.close()
            return None, None

        else:
            logging.info("Extracting audio and saving to '{}'".format(os.path.abspath(audio_path)))
            clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

        clip.close()

    return filename, audio_path


def merge_audio_video(video_path: str, audio_path: str, save_path: str):
    logging.info("Reading video file from '{}'".format(os.path.abspath(video_path)))
    video_clip = moviepy.VideoFileClip(video_path)

    logging.info("Reading audio file from '{}'".format(os.path.abspath(audio_path)))
    audio_clip = moviepy.AudioFileClip(audio_path)

    logging.info("Merging audio and video files...")
    final_clip = video_clip.with_audio(audio_clip)

    min_duration = min(video_clip.duration, audio_clip.duration)
    logging.info("Clipping result to {}".format(datetime.timedelta(seconds=min_duration)))
    final_clip = final_clip.subclipped(0, min_duration)

    logging.info("Writing result to '{}'".format(os.path.abspath(save_path)))
    final_clip.write_videofile(save_path, codec="libx264", audio_codec="aac")

    logging.info("Closing file streams")
    video_clip.close()
    audio_clip.close()
    final_clip.close()

import logging
import numpy as np

import yt_dlp
import librosa
import demucs.separate
from pydub import AudioSegment

def YoutubeDownload(youtube_url, output_path="downloaded.mp3"):
    print("\033[0m",end="")

    logging.info("downloading youtube audio ...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path.split(".")[0]}',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url.split("&")[0]])

    audio = AudioSegment.from_file(output_path,format="mp3")
    audio_duration = len(audio) / 1000
    return audio_duration
        

def ExtractAudioFeatures(file_path, n_mfcc=20):

    logging.info("extracting audio features ...")
    y, sr = librosa.load(file_path, sr=None)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features_dict = {
        "chroma_stft_mean": np.mean(chroma_stft),
        "chroma_stft_var": np.var(chroma_stft),
        "rms_mean": np.mean(rms),
        "rms_var": np.var(rms),
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_centroid_var": np.var(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_bandwidth_var": np.var(spectral_bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_var": np.var(rolloff),
        "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
        "zero_crossing_rate_var": np.var(zero_crossing_rate),
        "harmony_mean": np.mean(harmony),
        "harmony_var": np.var(harmony),
        "perceptr_mean": np.mean(perceptr),
        "perceptr_var": np.var(perceptr),
        "tempo": tempo
    }
    for i in range(n_mfcc):
        features_dict[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features_dict[f"mfcc{i+1}_var"] = np.var(mfccs[i])
    
    features = [float(features_dict[i].item()) if isinstance(features_dict[i], np.ndarray) 
                else float(features_dict[i]) for i in features_dict]
    
    return features

def AudioCutting(file_path, start_sec, sec=30, output_path="downloaded.mp3"):
    logging.info(f"cutting audio into {sec} seconds...")
    audio = AudioSegment.from_file(file_path,format="mp3")
    # 1s = 1000 ms
    start_time_ms = start_sec * 1000
    end_time_ms = (start_sec+sec) * 1000
    audio = audio[start_time_ms:end_time_ms]
    audio.export(output_path,format="mp3")

def AudioSeparate(file_path):
    logging.info("separating audio ...")
    # outputpath : separated/audio_separated
    demucs.separate.main(["--mp3", "--two-stems", "drums","-n", "audio_separated", file_path])
    demucs.separate.main(["--mp3", "--two-stems", "bass","-n", "audio_separated", file_path])
    demucs.separate.main(["--mp3", "--two-stems", "vocals","-n", "audio_separated", file_path])
    demucs.separate.main(["--mp3", "--two-stems", "other","-n", "audio_separated", file_path])

def mollifier(lst, time=0):
    if time > 0:
        n = len(lst)
        result = []
        for i in range(n):
            left = max(0, i - 1)
            right = min(n, i + 1 + 1) 
            window = lst[left:right]
            avg = sum(window) / len(window)
            result.append(avg)
        return mollifier(result, time-1)
    else: return lst
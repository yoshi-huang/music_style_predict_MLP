import librosa
import numpy as np
import os
import csv
from multiprocessing import Pool
from tqdm.rich import tqdm

def extract_audio_features(pack):

    try:
        file_path, music_style = pack
        n_mfcc=20

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

        features = {
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
            features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
            features[f"mfcc{i+1}_var"] = np.var(mfccs[i])

        file = file_path.split("\\")[1]
        print(file,"Done!")
        f = [file,661794]+[float(features[i].item()) if isinstance(features[i], np.ndarray) 
                    else float(features[i]) for i in features]+[music_style]

        return f
    
    except:
        print(pack[1],"wrong")

if __name__ == "__main__":

    music_style = input("input music_style you want to add > ")

    directory = music_style
    output_file = music_style+r"_features.csv"
    features = []
    for root, dirs, files in os.walk(directory): continue

    features = []
    pack = []
    for file in tqdm(files): pack.append([directory+"\\"+file,music_style])

    with Pool() as pool:
        features = pool.map(extract_audio_features,pack)
        
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer_test = csv.writer(outfile, delimiter=',')
        writer_test.writerows(features)

    print("\nComplete!")
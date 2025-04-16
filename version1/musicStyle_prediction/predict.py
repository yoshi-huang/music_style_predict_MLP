import numpy as np
import csv
import math
import os
import json
import librosa
import yt_dlp
from multiprocessing import Pool
import matplotlib.pyplot as plt
from pydub import AudioSegment

def cnvert_wav(file_path):

    audio = AudioSegment.from_file(file_path,format="mp3")
    file_path = file_path.split(".")[0]
    path = file_path+".wav"
    audio.export(path,format="wav")

    duration = len(audio) // 1000

    return path, duration

def cut_audio(file_path, output_path, start_sec):

    audio = AudioSegment.from_file(file_path,format="wav")
    # 1s = 1000 ms
    start_time_ms = start_sec * 1000
    end_time_ms = (start_sec+5) * 1000
    audio = audio[start_time_ms:end_time_ms]
    audio.export(output_path,format="wav")

def extract_audio_features(file_path, n_mfcc=20):

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

    return features

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(input,model_params):
    with open(model_params, "r") as f:
        data = json.load(f)
        weights = [np.array(w) for w in data["weights"]]
        biases = [np.array(b) for b in data["biases"]]

    x = input.astype(float)
    for i in range(len(weights)):
        z = np.dot(x, weights[i]) + biases[i]
        x = np.maximum(0, z) if i < len(weights)-1 else softmax(z)
    
    return x[0]

def yt_download(youtube_url, output_path):
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url.split("&")[0]])

def predict_part(pack):
    wav_path,magnitude,start_sec = pack

    i = start_sec
    segment_path = r"music_downloaded\segment_audio"+str(i)+r".wav"
    cut_audio(wav_path,segment_path,start_sec=i)

    features_dict = extract_audio_features(segment_path)
    features = [float(features_dict[i].item()) if isinstance(features_dict[i], np.ndarray) 
                else float(features_dict[i]) for i in features_dict]
    features = features * np.array(magnitude)

    model_params = r"data\model_params.json"
    segment_poss = forward(features,model_params)
    os.remove(segment_path)
    print(f"{i//60: >1}:{i%60:0>2} ~ {(i+5)//60: >1}:{(i+5)%60:0>2} Complete!  ",end="\r")
    return segment_poss

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

def style_save(possibe,genres):
    path = r"music_style_predicted.csv"
    with open(path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows([genres]+possibe)

def main():

    mp3_path = r"music_downloaded\target_audio"
    core = "All"
    mollifier_times = 8

    while True :
        yt_path = input("\033[0minput youtube url > \033[36m")    
        if yt_path=="\\core":
            core = input("\033[34mHow many CPU cores do you need for extract features > \033[0m")
        elif yt_path=="\\mollifier":
            mollifier_times = int(input("\033[34mHow many times do you need to use the mollifier > \033[0m"))
        else: break

    print("\033[0m",end="")
    
    yt_download(yt_path, mp3_path)
    print("\33[?25l\033[36mdownloading audio: Complete!\33[0m")

    mp3_path = r"music_downloaded\target_audio.mp3"
    wav_path, audio_time = cnvert_wav(mp3_path)
    print("\33[?25l\033[36mcnverting audio : Complete!\33[0m")

    features_path = r"data\features_30_sec.csv"
    with open(features_path, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=',')
        data = list(reader)[1]
        del data[0]
        del data[0]
        del data[-1]

        magnitude = []
        for d in data:
            d = abs(float(d))
            magnitude.append( 10**(-math.ceil((math.log10(d)))) )

    genres = ["blues", "classical", "country", "disco", "hiphop", 
            "jazz", "metal", "pop", "reggae", "rock", "lofi"]

    pack_init = [wav_path,magnitude]
    pack = [pack_init+[i] for i in range(0,audio_time-5,2)]
    
    print(f"\33[?25lextracting features ...\33[0m",end="\r")
    if core.isdigit() == True:
        with Pool(int(core)) as pool:
            possible = pool.map(predict_part,pack)
    else:
        with Pool() as pool:
            possible = pool.map(predict_part,pack)
    print(f"\33[?25l\033[36mextracting features : Complete!\33[0m")

    p_music = np.mean(possible,0)
    confidence = 100*float(np.max(p_music))
    genres_predict = genres[np.argmax(p_music)]

    print(f"\033[35mThere's a {confidence:.2f}% chance it's \033[33m{genres_predict}\033[35m music.")
    print("\033[?25h\033[0m",end="")
    os.remove(wav_path)

    colors = ['dodgerblue','orange','peru','y','yellowgreen',
              'steelblue','slategray','darkviolet','green','firebrick','fuchsia']
    legend = []
    
    plt.figure(figsize=(10,4))
    genres_p = np.array(possible).T
    for g in range(np.size(genres_p,0)) :
        genres_p[g] = mollifier(genres_p[g],mollifier_times)
        if np.max(genres_p[g]) > 0.2 :
            plt.plot(100*(genres_p[g]), "-", color=colors[g], lw=1)
            legend.append(genres[g]) 

    plt.xlim(0,np.size(possible,0)-1)
    plt.xticks([i for i in range(0,np.size(possible,0),10)],
            [f"{(2*j)//60}:{(2*j)%60:0>2}" for j in range(0,np.size(possible,0),10)],fontsize=7)
    plt.ylim(0,100)
    plt.xlabel("time (s)")
    plt.ylabel("styles possibility (%)")
    plt.legend(tuple(legend),loc=1,fontsize="xx-small")
    plt.show()

    style_save(possible,genres)

if __name__ == "__main__":
    print("\033[34msetting : \\core for setting multiprocess CPU")
    print("          \\mollifier for setting times of mollifier")
    while True:
        main()
        if input() == "quit": break
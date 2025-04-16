from pydub import AudioSegment
import os
from tqdm.rich import tqdm
import random

def cnvert_wav(file_path):

    audio = AudioSegment.from_file(file_path,format="mp3")
    file_path = file_path.split(".")[0]
    path = music_style+"\\"+file_path.split("\\")[1]+".wav"
    audio.export(path,format="wav")

    duration = len(audio) // 1000

    return path, duration

def cut_audio(file_path, output_path, start_sec):

    audio = AudioSegment.from_file(file_path,format="wav")
    # 1s = 1000 ms
    start_time_ms = start_sec * 1000
    end_time_ms = (start_sec+30) * 1000
    audio = audio[start_time_ms:end_time_ms]
    audio.export(output_path,format="wav")

music_style = input("input music_style you want to add > ")
directory = music_style+r"_mp3"

for root, dirs, files in os.walk(directory): continue

s = 1
for file in tqdm(files):
    print(s)
    pth, _ = cnvert_wav(music_style+"_mp3\\"+file)
 
    sec = 10*random.randint(3,8)
    print(s,sec,sec+30)
    cut_audio(pth,f"{music_style}\\{music_style}_{s:0>5}.wav",sec)
    s+=1

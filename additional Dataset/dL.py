import yt_dlp
from tqdm.tk import tqdm

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
        ydl.download([youtube_url])

music_style = input("input music_style you want to add > ")
f = open(music_style+"_set.txt","r",encoding='UTF-8')
s = 1

for line in tqdm(f.readlines()):
    if s>0:
        path = f"{music_style}_mp3\\{music_style}_{s:0>5}"
        yt_download(line.split("&")[0],path)
    s+=1

import torch
from model import Model
import matplotlib.pyplot as plt
from multiprocessing import Pool

import logging
import training.log_basic as log_basic

import os
import numpy as np
from preprocess import YoutubeDownload, ExtractAudioFeatures, AudioCutting, mollifier

def PoolPredict(pack):
    path, i = pack

    model = Model()
    model.load_state_dict(torch.load(r"model_parms.pth",weights_only=True))
    model.eval()

    output_path = f"audio_save/{i}to{i+3}audio.mp3"

    AudioCutting(path, start_sec=i, sec=3, output_path=output_path)
    features = ExtractAudioFeatures(output_path)
    os.remove(output_path)

    magnitude = [1, 10, 1, 100, 0.0001, 1e-06, 0.0001, 1e-05, 0.0001, 
                1e-06, 10, 1000, 10000, 100, 100000, 100, 0.001, 0.001, 
                0.0001, 0.001, 0.001, 0.01, 0.001, 0.01, 0.001, 0.1, 
                0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
                0.01, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 
                0.1, 0.01, 1, 0.01, 0.1, 0.01, 1, 0.01, 0.1, 0.01, 0.1, 0.01]
    input = np.array([features * np.array(magnitude)])

    x = torch.tensor(input).float()
    pred = model(x)

    return [i.item() for i in 100*pred[0]]


if __name__ == "__main__": 
    log_basic.__init__(logging.WARNING)

    genres = ["blues", "classical", "country", "disco", "hiphop", 
            "jazz", "metal", "pop", "reggae", "rock"]
    colors = ['dodgerblue','orange','peru','y','yellowgreen',
              'steelblue','slategray','darkviolet','green','firebrick','fuchsia']
    
    path = r"audio_save/temp_audio.mp3"
    cut_sec = 3
    duration = YoutubeDownload(youtube_url=input("input youtube url > \033[36m"), output_path=path)

    # create contents
    packs = [(path,i) for i in range(0,int(duration))]

    # predict every seconds
    with Pool() as pool:
        possible = np.array(pool.map(PoolPredict, packs)).T
    
    # draw plot
    plt.figure(figsize=(10,4.5))
    for g in range(np.size(possible,0)):
        possible[g] = mollifier(possible[g],2)
        plt.plot(possible[g], "-", color=colors[g], lw=1)
        plt.fill_between([i for i in range(np.size(possible,1))], 
                         possible[g], 
                         color=colors[g], alpha=0.25, 
                         label=genres[g])

    plt.xlim(0,np.size(possible,1)-1)
    plt.xticks(
        [i for i in range(0,np.size(possible.T,0),10)],
        [f"{(j)//60}:{(j)%60:0>2}" for j in range(0,np.size(possible.T,0),10)],
        fontsize=7,
        rotation=45
        )
    plt.ylim(0,100)
    plt.xlabel("time (s)")
    plt.ylabel("styles possibility (%)")

    plt.legend(loc=1,fontsize="xx-small")
    plt.show()
    
    plt.show()
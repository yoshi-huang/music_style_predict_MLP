import torch
from model import Model
import numpy as np
from preprocess import YoutubeDownload, ExtractAudioFeatures, AudioCutting

def main():
    genres = ["blues", "classical", "country", "disco", "hiphop", 
            "jazz", "metal", "pop", "reggae", "rock"]
    magnitude = [1, 10, 1, 100, 0.0001, 1e-06, 0.0001, 1e-05, 0.0001, 
                1e-06, 10, 1000, 10000, 100, 100000, 100, 0.001, 0.001, 
                0.0001, 0.001, 0.001, 0.01, 0.001, 0.01, 0.001, 0.1, 
                0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
                0.01, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 
                0.1, 0.01, 1, 0.01, 0.1, 0.01, 1, 0.01, 0.1, 0.01, 0.1, 0.01]

    model = Model()
    model.load_state_dict(torch.load(r"model_parms.pth",weights_only=True))
    model.eval()

    path = r"audio_save/temp_audio.mp3"
    url = input("input youtube url > ")
    YoutubeDownload(url, output_path=path)
    # just classify 30 ~ 33 seconds
    AudioCutting(file_path=path,
                 start_sec=30, 
                 sec=3,
                 output_path=path)
    input = np.array([ExtractAudioFeatures(path) * np.array(magnitude)])

    x_test = torch.tensor(input).float()
    pred = model(x_test)
    print([i.item() for i in 100*pred[0]])
        
if __name__ == "__main__": main()
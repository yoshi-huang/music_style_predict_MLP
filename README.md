# music style predict MLP
###### 一個完全不使用pytorch或tensorflow，手搓的MLP多層感知機

## 大致架構
1. 預處理使用使用librosa函式庫分析音樂的以下特徵：
    * chroma_stft
    * rms
    * spectral_centroid
    * spectral_bandwidth
    * rolloff
    * zero_crossing_rate
    * harmony
    * perceptr
    * tempo
    * mfcc（0～20）
2. 將提取出的特徵取平均及方差
3. 將提平均及方差輸入多層感知機做訓練：
    * 57個輸入單元的輸入層
    * 包含2層64個神經元的隱藏層
    * 11個使用softmax做分類的輸出層

## How to predict：
1. 確認你有安裝以下函式庫：
    * csv
    * librosa
    * json
    * multiprocessing
    * numpy
    * pydub
    * yt_dlp
若沒有，請使用pip安裝：\
```pip install csv librosa json multiprocessing numpy pydub yt_dlp```
2. 執行 prediction 中的 predict.py 檔案

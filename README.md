# MUSIC GENRES PREDICT MLP
一個使用pytorch預測音樂風格的MLP

## 大致架構
1. 使用librosa函式庫分析音樂特徵：[具體提出特徵](/workspaces/music_style_predict_MLP/features.md)
2. 將提取出的特徵取平均及方差
3. 將提平均及方差輸入模型做訓練

## How to predict：
1. 確認你有安裝以下函式庫：
    * csv
    * librosa
    * json
    * multiprocessing
    * numpy
    * pydub
    * yt_dlp

若沒有，請使用pip安裝：

```pip install csv librosa json multiprocessing numpy pydub yt_dlp```

2. 執行 version2 中的 Full_audio_predict.py 檔案
import csv
import math
import random
import numpy as np

def process_csv(input_file, output_file1, output_file2):
    with open(input_file, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=',')
        data = list(reader)[1:]  # 讀取所有行並去除標題行

    for r in data:
        del r[0]
        del r[0]
        
    magnitude = []
    for d in data[0][:-1]:
        d = abs(float(d))
        magnitude.append( 10**(-math.ceil((math.log10(d)))) )
        
    filtered_data = []

    genres = ["blues", "classical", "country", "disco", "hiphop", 
          "jazz", "metal", "pop", "reggae", "rock", "lofi"]
    genre_index = {genre: i for i, genre in enumerate(genres)}
    g = [[] for i in range(len(genres))]

    for r in data:
        
        s = np.zeros(11)
    
        if r[-1] in genre_index:
            s[genre_index[r[-1]]] = 1

        row = np.array(r[:-1]).astype(float)
        row = np.append(row * magnitude,s)
        g[genre_index[r[-1]]].append(row)
    
    # 拆分為訓練集與測試集
    train_data, test_data = [], []
    for i in g:
        random.shuffle(i)
        split_idx = int(0.8 * len(i))
        train_data += i[:split_idx]
        test_data += i[split_idx:]
    
    # 打亂數據順序
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # 寫入訓練集
    with open(output_file1, mode='w', newline='', encoding='utf-8') as outfile_train:
        writer_train = csv.writer(outfile_train, delimiter=',')
        writer_train.writerows(train_data)
    
    # 寫入測試集
    with open(output_file2, mode='w', newline='', encoding='utf-8') as outfile_test:
        writer_test = csv.writer(outfile_test, delimiter=',')
        writer_test.writerows(test_data)

# 指定輸入與輸出檔案名稱
input_csv = r"data\features_30_sec.csv"  # 請替換為實際檔案名稱

outfile_train = r"data\training_set.csv"
outfile_test = r"data\valid_set.csv"

# 執行處理函式
process_csv(input_csv, outfile_train, outfile_test)
print("Done!")

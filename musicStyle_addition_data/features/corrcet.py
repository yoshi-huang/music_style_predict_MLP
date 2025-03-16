import csv

features_csv = r"features_30_sec.csv"

with open(features_csv, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter=',')
    
    rewrite = []
    for row in reader:
        if rewrite == [] :
            rewrite.append(row)
            continue 
        '''
        title = row[0].split(".")[0].split("_")
        title[1] = int(title[1])+99
        re_title = title[0] + f"_{title[1]:0>5}"+".wav"

        row[0] = re_title
        rewrite.append(row)
        '''
        title = row[0].split(".")
        if title[1] != "wav" :
            if title[1] == "mp3" :
                row[0] = title[0] + ".wav"
            else:
                row[0] = title[0] + f"_{title[1]:0>5}"+".wav"
        rewrite.append(row)

with open(features_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(rewrite)

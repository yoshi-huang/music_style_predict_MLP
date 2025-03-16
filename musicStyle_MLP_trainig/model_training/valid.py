import json
import csv
import numpy as np

def load_csv(path):
    input, output = [], []
    with open(path, 'r', newline='') as tr_data:
        for row in csv.reader(tr_data):

            output.append([float(i) for i in row[-11:] ])
            input.append([ float(i) for i in row[:-11] ])
    return np.array(input), np.array(output)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

model_params = r"data\model_params.json"


def model(weights,biases):

    valid_set = r"data\valid_set.csv"
    input, output = load_csv(valid_set)

    for i in range(len(weights)):
        act = [input]
        for i in range(len(weights)):
            z = np.dot(act[-1], weights[i]) + biases[i]
            if i < len(weights) - 1:
                act.append(np.maximum(0, z))  # ReLU
            else:
                act.append(softmax(z))  # Sigmoid for binary classification

    y = act[-1]
    y_hat = output

    return y , y_hat


def Loss(weights,biases):
    y , y_hat = model(weights,biases)
    return -np.mean(np.sum(y_hat * np.log(y), axis=1))

def Correct(y,y_hat,valid=0,weights=[],biases=[]):
    
    if valid == 1:
        y , y_hat = model(weights,biases)

    Correct_Rate = 0
    for i in np.stack((y,y_hat),1):
        if np.argmax(i[0]) == np.argmax(i[1]) :
            Correct_Rate += 1
    Correct_Rate /= np.size(y,0)

    return Correct_Rate
    

if __name__ == "__main__":
    
    with open(model_params, "r") as f:
        data = json.load(f)
        weights = [np.array(w) for w in data["weights"]]
        biases = [np.array(b) for b in data["biases"]]

    Loss(weights,biases)
    
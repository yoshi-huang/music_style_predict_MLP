import csv
import math
import numpy as np
import logging as log
import training.log_basic as log_basic
from tqdm.rich import trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def init():
    log_basic.__init__(log.INFO,r"training/logging.log")
    log.info("torch import complete")

    if (device := torch.device("cuda" if torch.cuda.is_available() else "cpu")) == "cuda":
        log.info(f"{torch.cuda.get_device_name(0)}")
        log.info(f"{torch.cuda.device_count()}")
    log.info(f"Using device: {device}")
    return device

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(57,240),
            nn.BatchNorm1d(240),
            nn.ReLU(),

            nn.Linear(240,240),
            nn.BatchNorm1d(240),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(240,240),
            nn.BatchNorm1d(240),
            nn.ReLU(),

            nn.Linear(240,240),
            nn.BatchNorm1d(240),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(240,240),
            nn.BatchNorm1d(240),
            nn.ReLU(),

            nn.Linear(240,10),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x

def plt_save(lossA=[], lossB=[], accA=[]):
    epoch_count = 10

    # accuracy plt
    ax1.set_xlim((0,epochs/epoch_count))
    ax1.set_ylim((0,100))
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy rate (%)")

    ax1.plot(accA, "b-", lw=1)
    ax1.axhline(y=60, c="m", ls="--", lw=0.5)
    ax1.axhline(y=80, c="y", ls="--", lw=0.5)

    # loss plt
    ax2.set_xlim((0,epochs/epoch_count))
    ax2.set_ylim((0,-math.log(1/10)))
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("Loss")

    ax2.plot(lossA, "r-", lossB, "b-", lw=1)
    ax2.axhline(0.5, c="m", ls="--", lw=0.5)
    ax2.axhline(1.0, c="m", ls="--", lw=0.5)
    ax2.legend(('training','valid'),loc=1)

    fig.canvas.draw()
    fig.canvas.flush_events()

def csv_loader(path):
    with open(path, "r", newline="", encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=",")
        data = list(reader)
        x,y = [], []
        for r in data:
            x.append(r[:-10])
            y.append(r[-10:])

    x_data = np.array(x).astype(np.float32)
    y_data = np.array(y).astype(np.float32)
    return torch.from_numpy(x_data).float(), torch.from_numpy(y_data).float()

def epoch(e):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        opt.zero_grad()
        y_pred = model(x_batch)
        log.debug("batch training : forward complete")

        loss = crit(y_pred, y_batch)
        log.debug("batch training : loss calculation complete")

        loss.backward()
        opt.step()
        log.debug("batch training : weight update complete")

        train_loss += loss.item()
    train_loss /= len(train_loader)

    if (e+1) % 10 == 0:
            model.eval()
            val_loss = 0
            correct, total = 0, 0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    y_pred = model(x_batch)
                    log.debug("valid batch : forward complete")

                    loss = crit(y_pred, y_batch)
                    val_loss += loss.item()
                    log.debug("valid batch : loss calculation complete")

                    preds = torch.argmax(y_pred, dim=1)
                    y_correct = torch.argmax(y_batch,dim=1)
                    correct += (preds == y_correct).sum().item()
                    total += y_batch.size(0)
            
            train_loss_his.append(train_loss)
            val_loss /= len(val_loader)
            val_loss_his.append(val_loss)
            val_acc = correct/total
            acc_his.append(100*val_acc)

            log.info(f"epoch      : {e+1:> 4} / {epochs}")
            log.info(f"train_loss : {train_loss_his[-1]:>.8f}")
            log.info(f"val_loss   : {val_loss_his[-1]:>.8f}")
            log.info(f"val_acc    : {acc_his[-1]:>3.2f}%")
            plt_save(train_loss_his, val_loss_his, acc_his)

if __name__ == "__main__":
    
    device = init()

    model = Model().to(device)
    log.info(f"model construct complete")

    epochs = 400
    batch_size = 1200

    path = r"training/dataset.csv"
    x_tensor, y_tensor = csv_loader(path)

    log.debug(f"x data shape : {x_tensor.shape}")
    log.debug(f"y data shape : {y_tensor.shape}")

    train_size = int(0.8*len(x_tensor))
    val_size = (len(x_tensor) - train_size)
    train_data, val_data = random_split(TensorDataset(x_tensor.to(device),y_tensor.to(device)),[train_size,val_size])
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=True)
    log.info(f"data set import complete")

    crit = nn.CrossEntropyLoss()
    #opt = optim.Adadelta(model.parameters(),lr=0.5)
    opt = optim.Adam(model.parameters(),lr=1e-4)

    acc_his = []
    val_loss_his = []
    train_loss_his = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    plt_save(train_loss_his, val_loss_his, acc_his)
    
    for e in trange(epochs): epoch(e)
        
    log.info("training complete")

    plt.savefig(r"training/training_plot.png")
    plt.show()
    plt.ioff()
    log.info("figure drawing complete")

    # precision
    model.eval()
    precision = np.zeros(10)
    total = np.zeros(10)
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_pred = model(x_batch)
            log.debug("precision batch : forward complete")

            loss = crit(y_pred, y_batch)
            log.debug("precision batch : loss calculation complete")

            preds = torch.argmax(y_pred, dim=1)
            y_correct = torch.argmax(y_batch,dim=1)

            for index in range(len(y_correct)):
                precision[y_correct[index]] += (preds[index] == y_correct[index])
                total[y_correct[index]] += 1
            
    log.info(f"model precision : ")
    genres = ["blues", "classical", "country", "disco", "hiphop", 
            "jazz", "metal", "pop", "reggae", "rock"]
    precision = np.around(100*precision/total,2)
    for index in range(len(genres)):
        log.info(f"    {genres[index]: >9}: {precision[index]}%")
    
    if input("do you want to save this model : ")=="":  
        torch.save(model.state_dict(),r"model_parms.pth") 
        log.info("model save complete")
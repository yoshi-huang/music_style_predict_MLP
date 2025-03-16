import math
import numpy as np
from tqdm.tk import trange
import json
import matplotlib.pyplot as plt
import valid

class Model :

    def __init__(self, input_size, hidden_layer, output_size, learning_rate):
        self.learning_rate = learning_rate
        self.layers  = [input_size] + hidden_layer + [output_size]
        self.weights = []
        self.biases  = []
        self.act     = []
        self.pre_act = []
        self.d_act   = []

        for i in range(len(self.layers)-1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]) * 0.09)
            self.biases.append(np.ones([1, self.layers[i+1]]))

    def forward(self, input):
        self.act = [input]
        self.pre_act = []

        for i in range(len(self.weights)):
            z = np.dot(self.act[-1], self.weights[i]) + self.biases[i]
            self.pre_act.append(z)
            if i < len(self.weights) - 1:
                self.act.append(np.maximum(0, z))  # ReLU
            else:
                self.act.append(softmax(z))
        return self.act[-1]

    def loss(self, y_train):
        y_pred = self.act[-1]
        return -np.mean(np.sum(y_train * np.log(y_pred + 1e-10), axis=1))

    def optimize(self, y_train, λ):
        y_pred = self.act[-1]
        dL = (y_pred - y_train) / y_train.shape[0]

        self.d_act = [dL]

        for j in reversed(range(len(self.weights))):

            dz = self.d_act[-1]
            if j < len(self.weights) - 1:
                dz *= (self.pre_act[j] > 0)

            dw = np.dot(self.act[j].T, dz) + λ * self.weights[j]
            db = np.sum(dz, axis=0, keepdims=True)
            self.d_act.append(np.dot(dz, self.weights[j].T))

            dw = np.clip(dw, -clip_value, clip_value)
            db = np.clip(db, -clip_value, clip_value)
            
            self.weights[j] -= self.learning_rate * dw
            self.biases[j] -= self.learning_rate * db


def parameters_save(model, path):
    weights_list = [w.tolist() for w in model.weights]
    biases_list = [b.tolist() for b in model.biases]
    with open(path, "w") as f:
        json.dump({"weights": weights_list, "biases": biases_list}, f)

def plt_save(lossA=[], lossB=[], accA=[], accB=[]):

    # accuracy plt
    plt.subplot(1,2,1)
    plt.xlim((0,epochs/80))
    plt.ylim((0,100))
    plt.plot(accA, "r-", accB, "b-", lw=1)

    plt.xlabel("epochs")
    plt.ylabel("accuracy rate (%)")

    plt.axhline(y=60, c="m", ls="--", lw=0.5)
    plt.axhline(y=70, c="y", ls="--", lw=0.5)

    # loss plt
    plt.subplot(1,2,2)
    plt.xlim((0,epochs/80))
    plt.ylim((0,math.log2(output_size)))
    lossA = [math.log(i) for i in lossA] if log_plot else lossA
    lossB = [math.log(i) for i in lossB] if log_plot else lossB
    plt.plot(lossA, "r-", lossB, "b-", lw=1)

    plt.xlabel("epochs")
    plt.ylabel("log ( Loss )" if log_plot==1 else "Loss")

    plt.axhline(y=math.log(0.5) if log_plot else 0.5, c="m", ls="--", lw=0.5)
    plt.axhline(y=math.log(1.5) if log_plot else 1.5, c="m", ls="--", lw=0.5)

    plt.legend(('training','valid'),loc=1)

    plt.tight_layout()
    
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

if __name__ == "__main__":

    training_set  = r"data\training_set.csv"
    plt_path = r"model_training\acc_loss_plot.png"
    model_params  = r"data\model_params.json"

    lr = 1e-1
    clip_value = 6e-3
    L2_λ = 6e-3

    epochs = 7800
    batch_size = None

    input_size = 57
    hidden_layer = [64,64]
    output_size = 11

    log_plot = False
    ISplot = False

    x_train, y_train = valid.load_csv( training_set )

    model = Model(input_size, hidden_layer, output_size, lr)
    model.forward(x_train)

    '''
    x_batchs = np.split( x_train, [i for i in range(batch_size,np.size(x_train,0),batch_size)] )
    y_batchs = np.split( y_train, [i for i in range(batch_size,np.size(y_train,0),batch_size)] )
    '''

    train_loss_plt = [model.loss(y_train)]
    valid_loss_plt = [valid.Loss(model.weights,model.biases)]
    train_acc_plt = [0]
    valid_acc_plt = [0]

    if ISplot : 
        plt.ion()
        plt.figure(figsize=(10,4.2))

    for epoch in trange(epochs):

        '''
        for i in range(np.size(x_train,0)//batch_size+1):

            model_output = model.forward(x_batchs[i])
            model.optimize(y_batchs[i], L2_λ)
        '''

        model_output = model.forward(x_train)
        model.optimize(y_train, L2_λ)

        if epoch == 0 : print("\33[s\33[K",end="")

        if epoch % 80 == 0:   
            
            model_output = model.forward(x_train)
            train_loss_plt.append( model.loss(y_train) )
            train_acc_plt.append(100*valid.Correct(model_output,y_train))
            valid_loss_plt.append( valid.Loss(model.weights,model.biases))
            valid_acc_plt.append(100*valid.Correct([],[],1,model.weights,model.biases))
            
            print("\33[u",end="")
            
            print(f"\33[?25l\33[0mtrain loss : {train_loss_plt[-1]:.16f}" , 
                  "\033[31m⭡ " if train_loss_plt[-1]>train_loss_plt[-2] else "⭣ " )    
            print(f"\33[?25l\33[0mtrain accuracy : {train_acc_plt[-1]:.2f}%" , 
                  "⭡ " if train_acc_plt[-1]>train_acc_plt[-2] else "\033[31m⭣ " )
            
            print(f"\33[?25l\33[0mvalid loss : {valid_loss_plt[-1]:.16f}" , 
                  "\033[31m⭡ " if valid_loss_plt[-1]>valid_loss_plt[-2] else "⭣ " )    
            print(f"\33[?25l\33[0mvalid accuracy : {valid_acc_plt[-1]:.2f}%" , 
                  "⭡ " if valid_acc_plt[-1]>valid_acc_plt[-2] else "\033[31m⭣ " )
            
            if ISplot :
                plt.clf()
                plt_save(train_loss_plt, valid_loss_plt, train_acc_plt, valid_acc_plt)        
                plt.draw()

    valid_loss_plt.append(valid.Loss(model.weights,model.biases))

    if valid_loss_plt[-1] - np.min(valid_loss_plt) > 0.08:
        print("\033[33mOverfittingWarning: The final testing loss is not at its minimum, which might cause an overfitting problem.")
    print(f"\033[0mValid minimum is at {80*np.argmin(valid_loss_plt)-80} epoch.")
    
    plt.figure(figsize=(10,4.2))
    plt_save(train_loss_plt, valid_loss_plt, train_acc_plt, valid_acc_plt)
    plt.savefig(plt_path)
    plt.show()
    
    if input("Do you want to save the parameters (y or n) : ") != "n":
        parameters_save(model, model_params)
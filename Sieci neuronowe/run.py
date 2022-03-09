#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:51:50 2021

@author: RafaĹ Biedrzycki
Kodu tego mogÄ uĹźywaÄ moi studenci na Äwiczeniach z przedmiotu WstÄp do Sztucznej Inteligencji.
Kod ten powstaĹ aby przyspieszyÄ i uĹatwiÄ pracÄ studentĂłw, aby mogli skupiÄ siÄ na algorytmach sztucznej inteligencji. 
Kod nie jest wzorem dobrej jakoĹci programowania w Pythonie, nie jest rĂłwnieĹź wzorem programowania obiektowego, moĹźe zawieraÄ bĹÄdy.

Nie ma obowiÄzku uĹźywania tego kodu.
"""

import numpy as np
import pandas as pd
from scipy.special import expit

#ToDo tu prosze podac pierwsze cyfry numerow indeksow
p = [3, 7]

L_BOUND = -5
U_BOUND = 5
SHOW_EVERY = 10**3 

def q(x):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

x = np.linspace(L_BOUND, U_BOUND, 500)
y = np.array(q(x))
df = pd.DataFrame({'x': x, 'y': y})

np.random.seed(1)


# f logistyczna jako przykĹad sigmoidalej
def sigmoid(x):
    return 1/(1+np.exp(-x))

#pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)

     
#f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2

#pochodna f. straty
def d_nloss(y_out, y):
    return 2*( y_out - y )
    
class DlNet:
    def __init__(self, HIDDEN_SIZE, LR=0.003):
        self.y_out = 0
        
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.LR = LR
        self.hidden_params = np.random.normal(size=(self.HIDDEN_SIZE, 2)) * 10
        self.output_params = np.random.normal(size=(self.HIDDEN_SIZE)) * 10
        self.output_bias = np.random.normal() * 10

        self.y1 = np.zeros((self.HIDDEN_SIZE, 1))
        self.derivative_output_params = np.zeros(self.HIDDEN_SIZE)
        self.derivative_output_bias = 0
        self.derivative_hidden_params = np.zeros((self.HIDDEN_SIZE, 2))

        self.iter_in_minibatch = 0
#ToDo        

    
    def forward(self, x):  
        self.y1 = sigmoid(self.hidden_params[:, 0] * x + self.hidden_params[:, 1])
        self.y_out = sum(self.output_params * self.y1) + self.output_bias
#ToDo        
        
    def predict(self, x: float) -> float:    
        y1 = sigmoid(self.hidden_params[:, 0] * x + self.hidden_params[:, 1])
        return sum(self.output_params * y1) + self.output_bias
        
    def backward(self, x, y, batch_size):
        # output layer derivatives
        self.derivative_output_params += self.y1 * 2 * (self.y_out - y)
        self.derivative_output_bias += 2 * (self.y_out - y)
        
        # hidden layer derivatives
        dq_dy1 = 2 * (self.y_out - y) * self.output_params
        dq_ds = dq_dy1 * d_sigmoid(self.y1)
        self.derivative_hidden_params[:, 0] += dq_ds * x

        dq_dy1 = 2 * (self.y_out - y)
        dq_ds = dq_dy1 * d_sigmoid(self.y1)
        self.derivative_hidden_params[:, 1] += dq_ds#  * x

        self.iter_in_minibatch += 1

        if self.iter_in_minibatch == batch_size:
            self.hidden_params -= self.derivative_hidden_params * self.LR / batch_size
            self.output_bias -= self.derivative_output_bias * self.LR / batch_size
            self.output_params -= self.derivative_output_params * self.LR / batch_size

            self.derivative_output_params = np.zeros(self.HIDDEN_SIZE)
            self.derivative_output_bias = 0
            self.derivative_hidden_params = np.zeros((self.HIDDEN_SIZE, 2))

            self.iter_in_minibatch = 0


#ToDo        
        
    def train(self, df, iters, batch_size):
        loss = 0
        for epoch in range(0, iters):
            df = df.sample(frac = 1)
            x_set = list(df['x'])
            y_set = list(df['y'])
            for idx in range(len(x_set)):
                self.forward(x_set[idx])
                self.backward(x_set[idx], y_set[idx], batch_size)
                loss += nloss(self.y_out, y_set[idx])
            if epoch and not epoch % SHOW_EVERY:
                print('Epoch: ', epoch, 'Loss', loss/SHOW_EVERY)
                loss = 0

#ToDo                
    
nn = DlNet(HIDDEN_SIZE=100, LR=0.01)
nn.train(df, 20000, 5)

yh = [nn.predict(x_val) for x_val in x] #ToDo tu umiesciÄ wyniki (y) z sieci
# print(nn.output_params, nn.hidden_params)
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(x,y, 'r', label='train')
plt.plot(x,yh, 'b', label='test')
plt.legend()
plt.show()

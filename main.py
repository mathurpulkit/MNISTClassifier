import numpy as np
#import cupy as np
import readdata # self made library
import matplotlib.pyplot as plt
import time
import reportgen as rg


#Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def relu_back(x):
    return np.where(x>=0,np.ones(x.shape),np.zeros(x.shape))


def sigmoid_back(x):
    return np.exp(x)/((1+np.exp(x))**2)


def leakyrelu(x):
    return np.where(x > 0, x, x * 0.01)
    #where function uses operation 1 for true condition, 2nd for false


def leakyrelu_back(x):
    return np.where(x >= 0, 1, 0.01)


buffer = 10**-9
learning_rate = 0.05
num_iter = 1000


def initialise_parameters(network):
    parameters = {} # empty dictionary
    np.random.seed(69)
    for i in range(1,len(network)): # initialises weights and biases
        w_temp = np.random.randn(network[i], network[i-1])*0.1
        b_temp = np.zeros((network[i],1))
        print("W_temp size is" + str(w_temp.shape) + "\n")
        print("b_temp size is" + str(b_temp.shape) + "\n")
        parameters["W"+str(i)] = w_temp
        parameters["b"+str(i)] = b_temp
        #assert w_temp.shape == (network[i],network[i-1])
        #assert b_temp.shape == (network[i],1)
    return parameters


def forward_prop(A0, parameters):
    iter = len(parameters)//2 # no of weight arrays, i.e hidden layers+1
    A_temp = A0
    cache = {"A0": A0} # contains 2n+1 keys, n is no of layers
    for i in range(iter):
        Z_temp = np.dot(parameters["W"+str(i+1)], A_temp) + parameters["b"+str(i+1)]
        if i == iter - 1:  # last layer uses sigmoid
            A_temp = sigmoid(Z_temp)
        else:
            A_temp = leakyrelu(Z_temp)
        cache["Z" + str(i + 1)] = Z_temp
        cache["A" + str(i + 1)] = A_temp
    return A_temp, cache


def calc_cost(AL, Y_real, lamda = 0, parameters=None):
    # lamda and parameters are for future functionality
    m = Y_real.shape[1]
    #print("m=", m)
    cost = (-1/m)*(np.sum(np.multiply(Y_real,np.log(AL+buffer))+np.multiply(1-Y_real,np.log(1-AL+buffer))))
    #cost function, buffer added to avoid log(0) error
    #if (lamda):
    #    for i in range(len(parameters//2)):
    #        cost += (-1/(2*m))*lamda*np.sum(parameters["W"+str(i+1)])
    cost = np.squeeze(cost)
    return cost


def backprop(AL,Y,caches, parameters):
    grads = {}
    L = len(caches)//2
    m = AL.shape[1]
    #Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL+buffer) - np.divide(1 - Y, 1 - AL + buffer))
    dZ = dAL*sigmoid_back(caches["Z"+str(L)])
    dW = (1 / m) * np.dot(dZ, caches["A"+str(L-1)].T)
    db = (1 / m) * np.sum(dZ, axis=1)
    dA_prev = np.dot(parameters["W" + str(L)].T, dZ)
    grads["dA" + str(L)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db.reshape(-1,1)
    for l in reversed(range(1,L)):
        dZ = dA_prev * leakyrelu_back(caches["Z" + str(l)])
        dW = (1 / m) * np.dot(dZ, caches["A" + str(l-1)].T)
        db = (1 / m) * np.sum(dZ, axis=1)
        dA_prev = np.dot(parameters["W" + str(l)].T, dZ)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db.reshape(-1,1)
    return grads


def update_parameters(parameters, grads):
    L = len(parameters)//2
    for i in range(L):
        parameters["W" + str(i + 1)] -= learning_rate*grads["dW"+str(i+1)]
        parameters["b" + str(i + 1)] -= learning_rate * grads["db" + str(i + 1)]
    return #as parameters are passed by reference, no need to return


def train_model(X, Y_real, parameters, imgname="test"):
    costs = []
    for i in range(num_iter):
        Y, cache = forward_prop(X, parameters)
        cost = calc_cost(Y, Y_real)
        costs.append(cost)
        grads = backprop(Y, Y_real, cache, parameters)
        update_parameters(parameters,grads)
        if i % 10 == 0:
            print(i, "cost: ", cost)
    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("No of iterations")
    plt.savefig(imgname + ".png") #saves image
    return costs[len(costs)-1]


def check_accuracy(X, Y_real, parameters):
    Y_ret, cache = forward_prop(X,parameters)
    m = Y_ret.shape[1]
    Y_real_max = np.squeeze(np.argmax(Y_real, axis=0))
    Y_ret_max = np.squeeze(np.argmax(Y_ret, axis=0))
    #print(str(Y_real_max.shape),"  ",str(Y_ret_max.shape))
    accuracy = 0
    for i in range(m):
        #print(Y_real_max[i], "  ", Y_ret_max[i])
        if Y_real_max[i] == Y_ret_max[i]:
            accuracy += 1
    print(accuracy)
    return accuracy/m


def check_accuracy_test(parameters ,ifile="t10k-images.idx3-ubyte", lfile="t10k-labels.idx1-ubyte"):
    X, Y_real, images = readdata.read_input(ifile, lfile)
    Y_ret, cache = forward_prop(X, parameters)
    m = Y_ret.shape[1]
    Y_real_max = np.squeeze(np.argmax(Y_real, axis=0))
    Y_ret_max = np.squeeze(np.argmax(Y_ret, axis=0))
    # print(str(Y_real_max.shape),"  ",str(Y_ret_max.shape))
    accuracy = 0
    for i in range(m):
        print(Y_real_max[i], "  ", Y_ret_max[i])
        if Y_real_max[i] == Y_ret_max[i]:
            accuracy += 1
    print(accuracy)
    return accuracy / m



def rancheck(images, X,parameters):
    Y_ret, cache = forward_prop(X, parameters)
    for i in range(10):
        x=input("Enter num: ")
        pic = np.asarray(images[int(x)].squeeze())
        plt.imshow(pic)
        plt.show()
        print(np.argmax(Y_ret,axis=0)[int(x)])


def main():
    report_name = "report"
    idrep = 10  # report number
    X, Y_real, images = readdata.read_input()
    network = (X.shape[0], 112, 28, 10)
    parameters = initialise_parameters(network)
    cost = train_model(X,Y_real,parameters,imgname=report_name+str(idrep))
    acc_train = check_accuracy(X,Y_real,parameters)
    acc_test = check_accuracy_test(parameters)
    print("Train set Accuracy is: ", acc_train*100, "%")
    print("Test set Accuracy is: ", acc_test*100, "%")
    #rancheck(images, X, parameters)
    notes = input("Any Special Notes to add in the report: ")
    rg.reportgen(report_name+str(idrep),network,num_iter,
                 learning_rate,acc_train,acc_test,cost,note=notes)

main()



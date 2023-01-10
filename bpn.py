# %%
from pandas import DataFrame, read_csv
from numpy import exp
from random import uniform
import matplotlib.pyplot as plt


# sigmoid function
def sigmoid(x: float, _lambda: float = 1.0) -> float:
    '''
    x: activation
    _lambda: gain scale factor, default 1.0
    '''
    z = exp(-_lambda*x)
    sig = 1 / (1 + z)
    return sig


# bpn training function
def train_XOR_bpn(data: DataFrame, learning_rate: float, momentum: float, max_epoch: int, error_tolarence: float) -> tuple:
    '''
    data: input data
    learning_rate: speed of learning
    momentum: momentum factor
    max_epoch: when epoch reach the value of the parameter, stop training
    error_tolarence: when epoch loss is lesser then this parameter, stop training
    '''
    # initialize and keep weights
    n01 = uniform(-1, 1)
    n11 = uniform(-1, 1)
    n21 = uniform(-1, 1)
    m01 = uniform(-1, 1)
    m11 = uniform(-1, 1)
    m21 = uniform(-1, 1)
    m02 = uniform(-1, 1)
    m12 = uniform(-1, 1)
    m22 = uniform(-1, 1)

    # loss function: square error
    # assign a list space for loss
    loss = []

    # start
    print("epoch\tinput\tdesired\tactual\t\t\tweights")
    epoch = 0
    while epoch < max_epoch:
        epoch += 1
        pattern_loss_acc = 0
        delta_weights = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # for each epoch
        for index in data.index:
            # for each training record
            record = data.loc[index].values
            x0 = 1  # bias
            x1 = record[0]  # input 1
            x2 = record[1]  # input 2
            y = record[2]   # desired output
            if y == 1:
                y = 0.9
            else:
                y = 0.1

            # hidden nureon m1
            # calculate sumproduct and transform to signal
            sumproduct = x0*m01+x1*m11+x2*m21
            z1 = sigmoid(sumproduct)

            # hidden nureon m2
            # calculate sumproduct and transform to signal
            sumproduct = x0*m02+x1*m12+x2*m22
            z2 = sigmoid(sumproduct)

            # output nureon n
            # calculate sumproduct and transform to signal
            z0 = 1
            sumproduct = z0*n01+z1*n11+z2*n21
            s = sigmoid(sumproduct)

            # calculate error and delta
            error = y - s
            delta_output = error*s*(1-s)
            delta_hidden1 = m11*delta_output*z1*(1-z1)
            delta_hidden2 = m21*delta_output*z2*(1-z2)

            # weight delta
            d_n01 = learning_rate*delta_output*z0 + momentum*delta_weights[0]
            d_n11 = learning_rate*delta_output*z1 + momentum*delta_weights[1]
            d_n21 = learning_rate*delta_output*z2 + momentum*delta_weights[2]
            d_m01 = learning_rate*delta_hidden1*x0 + momentum*delta_weights[3]
            d_m11 = learning_rate*delta_hidden1*x1 + momentum*delta_weights[4]
            d_m21 = learning_rate*delta_hidden1*x2 + momentum*delta_weights[5]
            d_m02 = learning_rate*delta_hidden2*x0 + momentum*delta_weights[6]
            d_m12 = learning_rate*delta_hidden2*x1 + momentum*delta_weights[7]
            d_m22 = learning_rate*delta_hidden2*x2 + momentum*delta_weights[8]

            # update weight delta
            delta_weights = [d_n01, d_n11, d_n21, d_m01,
                             d_m11, d_m21, d_m02, d_m12, d_m22]

            # adjust weight
            n01 += d_n01
            n11 += d_n11
            n21 += d_n21
            m01 += d_m01
            m11 += d_m11
            m21 += d_m21
            m02 += d_m02
            m12 += d_m12
            m22 += d_m22

            # keep weights
            current_weights = [n01, n11, n21, m01, m11, m21, m02, m12, m22]
            print(f"{epoch}\t{x1,x2}\t{y}\t{s}\t{current_weights}")

            # calculate the accumulation of pattern loss
            pattern_loss_acc += (error**2)/2

        # calculate epoch loss
        loss.append(pattern_loss_acc/len(data.index))

        # if the weights of a epoch are convergent then stop training
        if loss[-1] < error_tolarence or epoch == max_epoch:
            print("final weights:", current_weights)
            return current_weights, loss


# bpn perdicting function
def XOR_bpn_perdiction(data: DataFrame, weights: list) -> tuple:
    '''
    data: input data    
    weights: trained weights
    '''
    # set weights
    n01, n11, n21, m01, m11, m21, m02, m12, m22 = weights

    # assign a list space for outputs
    outputs = []

    # start perdiction
    for index in data.index:
        # for each testing record
        record = data.loc[index].values
        x0 = 1  # bias
        x1 = record[0]  # input 1
        x2 = record[1]  # input 2

        # hidden nureon n1
        # calculate sumproduct and transform to signal
        sumproduct = x0*m01+x1*m11+x2*m21
        z1 = sigmoid(sumproduct)

        # hidden nureon n2
        # calculate sumproduct and transform to signal
        sumproduct = x0*m02+x1*m12+x2*m22
        z2 = sigmoid(sumproduct)

        # output nureon m
        # calculate sumproduct and transform to signal
        z0 = 1
        sumproduct = z0*n01+z1*n11+z2*n21
        s = sigmoid(sumproduct)

        outputs.append(s)
        print(f"input:{x1,x2}, output:{s}")

    return outputs


# %%
# read training data
training_data = read_csv("XOR.csv")

final_weights, loss = train_XOR_bpn(training_data, 0.5, 0.1, 50000, 0.05)

# %%
epoch_list = [i for i in range(1, len(loss)+1)]
plt.plot(epoch_list, loss)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# %%
testing_data = training_data
XOR_bpn_perdiction(testing_data, final_weights)

# %%

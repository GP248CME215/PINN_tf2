import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):

        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]

        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]

        self.u = u

        self.layers = layers
        self.nu = nu

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # Create a list including all training variables
        self.train_variables = self.weights + self.biases
        # Key point: anything updates in train_variables will be
        #            automatically updated in the original tf.Variable

        # define the loss function
        self.loss = self.loss_NN()


    '''
    Functions used to establish the initial neural network
    ===============================================================
    '''

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases



    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)



    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y


    '''
    Functions used to building the physics-informed contrainst and loss
    ===============================================================
    '''

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u


    def net_f(self, x,t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u*u_x - self.nu*u_xx
        return f


    @tf.function
    # calculate the physics-informed loss function
    def loss_NN(self):
        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.f_pred = self.net_f(self.x_f, self.t_f)
        loss = tf.reduce_mean(tf.square(self.u - self.u_pred)) + \
               tf.reduce_mean(tf.square(self.f_pred))
        return loss


    def train(self, nIter: int, learning_rate: float):
        """Function used for training the model"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        varlist = self.weights + self.biases # all trainable parameters
        start_time = time.time()

        for it in range(nIter):
            optimizer.minimize(self.loss_NN, varlist)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss = self.loss_NN().numpy()
                print('It: %d, Train Loss: %.3e, Time: %.2f' % (it, loss, elapsed))
                start_time = time.time()

    '''
    Functions used to define L-BFGS optimizers
    ===============================================================
    '''
    @tf.function
    def predict(self, X_star):
        u_star = self.net_u(X_star[:,0:1], X_star[:,1:2])
        f_star = self.net_f(X_star[:,0:1], X_star[:,1:2])
        return u_star, f_star

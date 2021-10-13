"""
@author: Yongji Wang (modified from Maziar Raissi)
"""
import time

import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs

from pinn import PhysicsInformedNN

np.random.seed(123)
tf.random.set_seed(123)


if __name__ == "__main__":
    nu = 0.01/np.pi
    noise = 0.0

    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 1]

    data = scipy.io.loadmat("burgers_shock.mat")

    t = data["t"].flatten()[:,None]
    x = data["x"].flatten()[:,None]
    Exact = np.real(data["usol"]).T

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    X_u_train = tf.cast(X_u_train, dtype=tf.float32)
    u_train = tf.cast(u_train, dtype=tf.float32)
    X_f_train = tf.cast(X_f_train, dtype=tf.float32)

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)

    start_time = time.time()
    model.train(5000)
    elapsed = time.time() - start_time
    print("Training time: %.4f" % (elapsed))

    X_star = tf.cast(X_star, dtype=tf.float32)
    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print("Error u: %e" % (error_u))


    U_pred = griddata(X_star, u_pred.numpy().flatten(), (X, T), method="cubic")
    Error = np.abs(Exact - U_pred)

    data_to_save = {
        "X_u_train": X_u_train.numpy(),
        "X_f_train": X_f_train.numpy(),
        "u_train": u_train.numpy(),
        "U_pred": U_pred,
    }
    scipy.io.savemat("script_output.mat", data_to_save)

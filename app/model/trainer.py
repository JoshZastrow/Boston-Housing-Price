import numpy as np
from sklearn.externals import joblib
from keras.datasets import boston_housing
from sklearn import preprocessing as p
from sklearn import model_selection as mdl
import pickle
import dill
import sys

class NeuralNetwork():
    """
    Two (hidden) layer neural network model.
    First and second layer contain the same number of hidden units
    """

    def __init__(self, input_dim, units, std=0.0001):
        self.params = {}
        self.input_dim = input_dim

        self.params['W1'] = np.random.rand(self.input_dim, units)
        self.params['W1'] *= std
        self.params['b1'] = np.zeros((units))

        self.params['W2'] = np.random.rand(units, units)
        self.params['W2'] *= std * 10  # Compensate for vanishing gradients
        self.params['b2'] = np.zeros((units))

        self.params['W3'] = np.random.rand(units, 1)
        self.params['b3'] = np.zeros((1,))

    def mse_loss(self, x, y=None, drop_p=0.9, reg=0.01,
                 evaluate=False, predict=False):

        # Unpack variables
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = x.shape

        ###############################################
        # Forward Pass
        ###############################################
        Fx = None

        # First Layer
        x1 = np.dot(x, W1) + b1

        # Activation
        a1 = np.maximum(x1, 0)

        # Drop Out
        drop1 = np.random.choice([1, 0], size=x1.shape, p=[
                                 drop_p, 1 - drop_p]) / drop_p
        a1 *= drop1

        # Second Layer
        x2 = np.dot(a1, W2) + b2

        # Activation
        a2 = np.maximum(x2, 0)

        # Drop Out
        drop2 = np.random.choice([1, 0], size=x2.shape, p=[
                                 drop_p, 1 - drop_p]) / drop_p
        a2 *= drop2

        # Final Layer
        x3 = np.dot(a2, W3) + b3

        # Output
        Fx = x3

        if predict:
            return Fx

        # Mean Squared Error Cost Function
        mse_loss = np.sum((Fx - y.reshape(-1, 1))**2, axis=0) / N
        mae_loss = np.sum(np.absolute(Fx - y.reshape(-1, 1)), axis=0) / N
        wght_loss = 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**3))
        loss = mse_loss + wght_loss

        if evaluate:
            return {'loss': loss,
                    'mean_absolute_error': mae_loss[0],
                    'mean_squared_error': mse_loss[0],
                    'weight_loss': wght_loss}

        #############################################
        # Backpropagation
        #############################################

        grads = {}

        # Output
        dFx = 2 * (Fx.copy() - y) / N  # [50, 1]

        # Final Layer
        dx3 = np.dot(dFx, W3.T)
        dW3 = np.dot(x2.T, dFx)
        db3 = np.sum(dFx * N, axis=0)

        # Drop Out
        dx3 *= drop2

        # activation
        da2 = a2.copy()
        da2[da2 > 0] = 1
        da2[da2 <= 0] = 0
        da2 *= dx3

        # Second Layer
        dx2 = np.dot(da2, W2.T)
        dW2 = np.dot(x1.T, da2)
        db2 = np.sum(da2, axis=0)

        # Drop out
        dx2 *= drop1

        # activation
        da1 = a1.copy()
        da1[da1 > 0] = 1
        da1[da1 < 0] = 0
        da1 *= dx2

        # First Layer
        dx1 = np.dot(da1, W1.T)
        dW1 = np.dot(x.T, da1)
        db1 = np.sum(da1, axis=0)

        grads['W3'] = dW3
        grads['b3'] = db3
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1

        grads['W3'] += dW3 * reg
        grads['W2'] += dW2 * reg
        grads['W1'] += dW1 * reg

        return mae_loss, loss, grads

    def fit(self, X, y, validation_data, epochs=80,
            learning_rate=1e-3, learning_rate_decay=0.99,
            reg=1e-5, batch_size=50, dropout_val=0.95):

        assert type(validation_data) == tuple
        x_val, y_val = validation_data

        num_train = X.shape[0]
        iters_per_epoch = max(num_train // batch_size, 1)
        val_acc = 0

        loss_history = []
        val_loss_history = []
        mae_history = []
        val_mae_history = []

        for e in range(epochs):
            for it in range(iters_per_epoch):
                x_batch = None
                y_batch = None

                batch_indices = np.random.choice(num_train,
                                                 batch_size,
                                                 replace=False)

                x_batch = X[batch_indices]
                y_batch = y[batch_indices]

                mae, loss, grads = self.mse_loss(x_batch,
                                                 y_batch,
                                                 drop_p=dropout_val,
                                                 reg=reg)

                val_mae, val_loss, _ = self.mse_loss(x_val, y_val)

                for key in self.params:
                    self.params[key] -= learning_rate * grads[key]

                if it % iters_per_epoch == 0:
                    learning_rate *= learning_rate_decay

            # Record cost values for this epoch
            loss_history.append(loss)
            mae_history.append(mae)
            val_loss_history.append(val_loss)
            val_mae_history.append(val_mae)

        return {'loss': loss_history,
                'mean_absolute_error': mae_history,
                'val_loss': val_loss_history,
                'val_mean_absolute_error': val_mae_history}

    def evaluate(self, X, y):
        return self.mse_loss(X, y, drop_p=1, evaluate=True)

    def predict(self, X):
        return self.mse_loss(X, drop_p=1, predict=True)


if __name__ == "__main__":

    # Get Data
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    # join the x and y together for shuffling
    y_train = y_train.reshape((y_train.shape[0], 1))
    data = np.hstack([x_train, y_train])

    # Make random generation repeatable
    seed = 7
    np.random.seed(seed)

    # small dataset, shuffling and folding the data
    shuff_count, i = 3, 0
    split_count = 4
    epoch_count = 80
    b = 0

    # Fit a scaling function to the train data
    scale = p.StandardScaler().fit(x_train)

    # Initiate First Model
    best_model = NeuralNetwork(input_dim=x_train.shape[1], units=64)  # base_model()

    while i < shuff_count:
        i += 1
        data_folds = mdl.KFold(n_splits=split_count,
                               shuffle=True,
                               random_state=seed).split(data)

        for dfold in data_folds:
            train, valid = dfold

            # Seperate each k fold of train data into a train and validation set
            xt = scale.transform(data[train, :-1])
            yt = data[train, -1:]

            # Transform validation set based on scaling on training set
            xv = scale.transform(data[valid, :-1])
            yv = data[valid, -1:]

            model = NeuralNetwork(
                input_dim=x_train.shape[1], units=64)  # base_model()

            history = model.fit(xt, yt, validation_data=(xv, yv), epochs=epoch_count)

            # Evaluate each model, keep the best one
            curr_evaluation = model.evaluate(
                scale.transform(x_test), y_test)  # , verbose=0)
            best_evaluation = best_model.evaluate(
                scale.transform(x_test), y_test)  # , verbose=0)

            if curr_evaluation['mean_squared_error'] < \
               best_evaluation['mean_squared_error']:
                best_model = model

            b += 1

    print('Base Model mean absolute error on test results: $', end='')
    mae = best_model.evaluate(scale.transform(x_test), y_test[:, None])[
        'mean_absolute_error']
    print(round(mae * 1000, 2))

    # Dump Model
    dill.dump(best_model, open("app/model/model.pkl", "wb"))

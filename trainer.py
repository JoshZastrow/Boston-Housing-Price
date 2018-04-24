import numpy as np
from sklearn.externals import joblib
from keras.datasets import boston_housing
from sklearn import preprocessing as p
from sklearn import model_selection as mdl
from model import NeuralNetwork

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
    joblib.dump(best_model, 'model.pkl')

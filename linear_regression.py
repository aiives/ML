#----------------------------------------------------Description
"""
This is a simple linear regression example. It was heavily inspired by the example in Joel Grus' book Data Science from Scratch. I mostly just changed it to use numpy arrays instead of lists to improve performance.
"""
#----------------------------------------------------Imports
import numpy as np 
from typing import Tuple
import numpy.typing as npt
import matplotlib.pyplot as plt 
from sklearn.datasets import make_regression

#----------------------------------------------------Functions
def train_test_split( X: npt.NDArray, y: npt.NDArray, test_size=0.3, shuffle=True, seed=None) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Create a test and training set from Data
        -----Parameters
        X:      Data with shape (n_samples, n_features)
        y:      labels with shape (n_samples)
        test_size:  percentage of Data to be used in Testset
        shuffle:    if True: data is shuffled randomly
        seed:   possibility to set a seed for np.random
        -----Returns
        X_train:    Trainingdata with shape(n_samples*(1-test_size), n_features)
        X_test:     Testdata
        y_train:    Traininglabels
        y_test      Testlabels
    """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_ind = int(np.shape(y)[0] * test_size)
    X_test, X_train = X[:split_ind], X[split_ind:]
    y_test, y_train = y[:split_ind], y[split_ind:]
    return X_train, X_test, y_train, y_test

def shuffle_data(X: npt.NDArray, y: npt.NDArray, seed=None) -> Tuple[npt.NDArray, npt.NDArray]:
    """Shuffles datapoints randomly
    -----Parameters
    X:      Data with shape (n_samples, n_features)
    y:      labels with shape (n_samples)
    -----Returns
    X:      Same NDArray with shuffled rows (samples)
    y:      Shuffled labels 
    """
    if seed:
        np.random.seed(seed)
    ind = np.arange(X.shape[0])
    np.random.shuffle(ind)
    return X[ind], y[ind]

#----------------------------------------------------Classes
class Regression(object):
    """Base regression model
    -----Parameters
    n_iter:     number of times the weights will be updated
    learning_rate:  influences by how much weights will be tuned
    """
    def __init__(self, n_iter: int, learning_rate: float):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __initialize_weights(self, n_features: int) -> None:
        """initialize weights with random value between -1 and 1"""
        self.w = np.random.uniform(-0.1, 0.1, (n_features,))

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> None:
        """Starts with random weights (model parameters), calculates the predition_error for each iteration and then updates the weights accordingly
        -----Parameters
        X:  Inputdata with shape (n_samples, n_features)
        y:  Targetlabels for each sample
        """
        X = np.insert(X, 0, 1, axis=1) #add a column of 1s for bias weights
        self.error_history = []
        self.__initialize_weights(n_features=X.shape[1])
        for i in range(self.n_iter):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5 * (y - y_pred)**2) #add regularization?
            self.error_history.append(mse)
            grad = -(y - y_pred).dot(X)
            self.w -= self.learning_rate * grad

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Calculates predigted target values
        -----Parameters
        X:  Inputdata with shape (n_samples, n_features)
        -----Returns
        y_pred: Predicted labels calculated by multypling weights with features of X
        """
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.w)
        


class LinearRegression(Regression):
    def __init__(self, n_iter=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        super(LinearRegression, self).__init__(n_iter=n_iter, learning_rate=learning_rate)
    def fit(self, X, y):
        super(LinearRegression, self).fit(X, y)

#----------------------------------------------------Main
def main() -> None:
    #prepare a toy-dataset
    n_iter = 100
    X, y = make_regression(n_samples=1000, n_features =1, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

    #create, fit and predict with linear model
    model = LinearRegression(n_iter=n_iter,learning_rate=0.00005 )
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    #Visualize Data and Model Prediction
    figure, axs = plt.subplots(2)
    axs[0].scatter(X_test,y_test)
    axs[0].set_title("Test data and predicted regression line")
    axs[0].plot(X_test, y_pred, color='red')
    axs[1].plot(range(n_iter), model.error_history, color='red')
    axs[1].set_title("Model trainingerror over time (iterations)")
    #plt.show()

if __name__ == "__main__":
    main()

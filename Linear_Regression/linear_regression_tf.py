#! /usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def plot_decision_boundary(X, Y, model):
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 30)
    y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1, 30)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)


n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
              np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts), np.random.normal(6, 2, n_pts)]).T
X = np.vstack((Xa, Xb))
Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
adam = Adam(learning_rate=0.1)
model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x=X, y=Y, verbose=1, batch_size=50,
                    epochs=15, shuffle=True)

test_point_x = 2.5
test_point_y = 4
test_point = np.array([[test_point_x, test_point_y]])
prediction = model.predict(test_point)
print("Prediction is: ", prediction)

plot_decision_boundary(X, Y, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.title('Accuracy')
plt.show()

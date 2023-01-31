import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import math


def f(x, y):
    return (math.sqrt(math.sin(x))) + 2 * y

#losowanie 100 punktów do tablicy na której będziemy nauczać naszą sieć
X = np.random.rand(100, 2)
X[:, 0] = np.linspace(0, 3, 100)
print(X)
#obliczanie wartości funkcji dla każdej pary punktów
y_true = np.array([(f(x[0], x[1])) for x in X])
print(y_true)
#inicjalizacja sieci neuronowej
model = Sequential()
#definowanie warstw ukrytych i ilości neuronów, w pierwszej linii również informujemy o kształcie inputu (tablica 2-elem.)
model.add(Dense(units=5, input_shape=(2,)))
model.add(Dense(units=3))
model.add(Dense(units=1))
#kompulijemy sieć korzystając z metody liczenia strat mean_squared_error
model.compile(optimizer=RMSprop(), loss='mean_squared_error')
#ksztalcimy siec przez 100 epok po 5 razy.
history = model.fit(x=X, y=y_true, batch_size=5, epochs=100)
print(model.summary())
print(history.history['loss'])
#rysowanie wykresow
osx = [i for i in range(100)]
y_pred = model.predict(X)
plt.plot(osx, y_true, c='k', label="Prawdziwe wartosci funkcji")
plt.plot(osx, y_pred, c='r', label="Wartosci funkcji przewidziane przez siec neuronowa")
plt.title("")
plt.legend()
plt.show()
err = plt.figure(2)
plt.plot(history.history['loss'])
plt.title("Wykres funkcji straty (mean_squared_error)")
plt.xlabel("Liczba epok")
plt.ylabel("Wartość błędu")
err.show()

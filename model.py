from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout


def lstm_model():
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(seq_len, len(good_chars))))
    model.add(Dropout(0.4))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(len(good_chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

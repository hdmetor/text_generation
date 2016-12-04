from model import lstm_model
from load_data import get_data
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout


epochs = 100
batch_size = 128
seq_len = 100

checkpointer = ModelCheckpoint(
    filepath="weights.hdf5",
    verbose=1,
    save_best_only=True
)
print('Loading data')
X, Y = get_data(seq_len=seq_len)
seq_len, total_char = X.shape[1:]
model = lstm_model(seq_len, total_char)
print(X.shape)
print(Y.shape)



print('start training')
model.fit(X, Y, batch_size=batch_size, nb_epoch=epochs,  callbacks=[checkpointer])
print('training done')
model.save('final.hdf5')

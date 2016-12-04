from model import lstm_model
from load_data import get_data
from keras.callbacks import ModelCheckpoint

epochs = 100
batch_size = 128

checkpointer = ModelCheckpoint(
    filepath="weights.hdf5",
    verbose=1,
    save_best_only=True
)
print('Loading data')
X, Y = get_data()

print('start training')
model.fit(X, Y, batch_size=batch_size, nb_epoch=epochs,  callbacks=[checkpointer])
print('training done')
model.save('final.hdf5')

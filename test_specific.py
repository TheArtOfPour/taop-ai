from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Dropout, regularizers, \
    BatchNormalization, Convolution1D, Flatten, GaussianNoise, GaussianDropout
from keras.regularizers import l2
import keras
import numpy
import pickle

from keras.optimizers import SGD

numpy.random.seed()
style_dims = 15
yeast_dims = 15
hop_dims = 38
fermentable_dims = 39

verbose = 1
epochs = 75

yeast_data = numpy.array(pickle.load(open("data/one_hot_yeast.p", "rb")))
hops_data = numpy.array(pickle.load(open("data/one_hot_hops.p", "rb")))
hops_amounts = numpy.array(pickle.load(open("data/hop_amounts.p", "rb")))
hops_times = numpy.array(pickle.load(open("data/hop_times.p", "rb")))
fermentables_data = numpy.array(pickle.load(open("data/one_hot_fermentables.p", "rb")))
fermentables_amounts = numpy.array(pickle.load(open("data/fermentable_amounts.p", "rb")))
train_labels = numpy.array(pickle.load(open("data/labels.p", "rb")))
one_hot_labels = keras.utils.to_categorical(train_labels, num_classes=style_dims)
print(len(train_labels))

tb_cb = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=0,
    batch_size=32,
    write_graph=True,
    write_grads=False,
    write_images=False,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None
)


def fit_model(lr=0.001, dropout=0.55, reg=0.0004, yw=35, hw=75, fw=125, final=200):
    print("-----------------------1--------------------------")
    print("lr:" + str(lr) + " dropout:" + str(dropout) + " yw:" + str(yw) +
          " hw:" + str(hw) + " fw:" + str(fw) + " final:" + str(final))

    total_width = yw + hw + fw

    # yeast
    y = Input(shape=(yeast_dims,), name='yeast')
    y1 = Dense(yw, activation='relu')(y)
    y2 = Dropout(dropout, input_shape=(yw,))(y1)
    y3 = Dense(yw, activation='relu')(y2)

    # hops
    h = Input(shape=(9, hop_dims,), name='hop')
    conv_h = Convolution1D(32, (9,),
                           padding='same',
                           kernel_initializer='he_uniform',
                           kernel_regularizer=l2(0.0001),
                           activation='relu')(h)
    conv_h = BatchNormalization()(conv_h)
    h4 = Dense(hw, activation='relu', kernel_regularizer=regularizers.l2(reg))(conv_h)
    h5 = Dropout(dropout, input_shape=(hw,))(h4)
    h6 = Dense(hw, activation='relu', kernel_regularizer=regularizers.l2(reg))(h5)
    h7 = Flatten()(h6)

    # fermentables
    f = Input(shape=(5, fermentable_dims,), name='fermentable')
    conv_f = Convolution1D(32, (5,),
                           padding='same',
                           kernel_initializer='he_uniform',
                           kernel_regularizer=l2(0.0001),
                           activation='relu')(f)
    conv_f = BatchNormalization()(conv_f)
    f4 = Dense(fw, activation='relu', kernel_regularizer=regularizers.l2(reg))(conv_f)
    f5 = Dropout(dropout, input_shape=(fw,))(f4)
    f6 = Dense(fw, activation='relu', kernel_regularizer=regularizers.l2(reg))(f5)
    f7 = Flatten()(f6)

    # with our networks combined...
    a = Concatenate()([y3, h7, f7])
    a1 = Dense(total_width, activation='relu')(a)
    tn = GaussianNoise(0.01)(a1)
    t = Dense(total_width, activation='relu')(tn)
    td1 = GaussianDropout(dropout, input_shape=(total_width,))(t)
    a5 = Dense(final, activation='relu', kernel_regularizer=regularizers.l2(reg))(td1)
    out = Dense(style_dims, activation='softmax', name='style')(a5)
    tf_model = Model(inputs=[y, h, f], outputs=out)

    sgd = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False)
    tf_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    tf_model.fit(
        [yeast_data, hops_data, fermentables_data],
        [one_hot_labels],
        epochs=epochs, verbose=verbose, validation_split=0.1,
        callbacks=[tb_cb]
    )
    return tf_model


model = fit_model()

print("saving")
model.save("models/specific.h5")
model.save_weights("models/specific_weights.h5")

print("complete")

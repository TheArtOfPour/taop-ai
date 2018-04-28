from keras.models import Model
from keras.layers import Dense, Flatten, Input, Add, Concatenate, Dropout, regularizers
from keras.optimizers import SGD
import keras
import numpy
import pickle
# fix random seed for reproducibility
numpy.random.seed(7)

lr = 0.001

# @todo : capture temperature data (mash and ferm)
style_dims = 15
# shape 19
yeast_dims = 15
yeast_data = numpy.array(pickle.load(open("data/one_hot_yeast.p", "rb")))

# shape 89, 1, 120
hop_dims = 38
hops_data = numpy.array(pickle.load(open("data/one_hot_hops.p", "rb")))
hops_amounts = numpy.array(pickle.load(open("data/hop_amounts.p", "rb")))
hops_times = numpy.array(pickle.load(open("data/hop_times.p", "rb")))

# shape 64, 1, 65
fermentable_dims = 39
fermentables_data = numpy.array(pickle.load(open("data/one_hot_fermentables.p", "rb")))
fermentables_amounts = numpy.array(pickle.load(open("data/fermentable_amounts.p", "rb")))

train_labels = numpy.array(pickle.load(open("data/labels.p", "rb")))
one_hot_labels = keras.utils.to_categorical(train_labels, num_classes=style_dims)
print(len(train_labels))

# todo : try shallower categorical output model
# todo : balance training data

# make and merge separate
y = Input(shape=(yeast_dims,))
y1 = Dense(75, activation='relu')(y)
# y2 = Dense(75, activation='relu')(y1)
y3 = Dropout(0.5, input_shape=(75,))(y1)


h = Input(shape=(9, hop_dims,))
h1 = Dense(200, activation='relu')(h)
h2 = Flatten()(h1)
# h3 = Dense(200, activation='relu')(h2)

ha = Input(shape=(9,))
ha1 = Dense(25, activation='relu')(ha)

ht = Input(shape=(9,))
ht1 = Dense(25, activation='relu')(ht)

h_added = Concatenate()([h2, ha1, ht1])
h4 = Dense(250, activation='relu')(h_added)
h5 = Dropout(0.5, input_shape=(250,))(h4)


f = Input(shape=(5, fermentable_dims,))
f1 = Dense(200, activation='relu')(f)
f2 = Flatten()(f1)
# f3 = Dense(200, activation='relu')(f2)

fa = Input(shape=(5,))
fa1 = Dense(25, activation='relu')(fa)

f_added = Concatenate()([f2, fa1])
f4 = Dense(225, activation='relu')(f_added)
f5 = Dropout(0.5, input_shape=(225,))(f4)


a = Concatenate()([y3, h5, f5])
a1 = Dense(550, activation='relu')(a)
a2 = Dropout(0.5, input_shape=(550,))(a1)
a3 = Dense(750, activation='relu')(a2)
out = Dense(style_dims)(a3)
model = Model(inputs=[y, h, ha, ht, f, fa], outputs=out)

print("compiling")
sgd = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False)
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("fitting")
model.fit(
    [yeast_data, hops_data, hops_amounts, hops_times, fermentables_data, fermentables_amounts],
    [one_hot_labels],
    epochs=75, verbose=1, validation_split=0.1
)

print("saving")
model.save("models/regal.h5")
model.save_weights("models/regal_weights.h5")

print("complete")

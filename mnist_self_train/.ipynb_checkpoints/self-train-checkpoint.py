'''
mnist 半监督学习
'''

import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Input, Flatten, MaxPooling2D, Dropout
from keras.datasets import mnist
from keras.optimizers import Adam

# Load dataset
(xtr,ytr),(xte,yte) = mnist.load_data()
xtr = np.expand_dims(xtr,3).astype('float32')/255.
ytr = keras.utils.to_categorical(ytr, 10)
xte = np.expand_dims(xte,3).astype('float32')/255.
yte = keras.utils.to_categorical(yte, 10)
perm = np.load('perm-60000.npy')

# 100 label samples, 59900 unlabel samples
xla = xtr[perm[0:100]]
yla = ytr[perm[0:100]]
xun = xtr[perm[100:]]
yun = ytr[perm[100:]]

# Build Model
model = Sequential([
    Conv2D(64, (3,3), padding='same', activation='relu', name='conv1', input_shape=(28,28,1)),
    MaxPooling2D((2,2), name='pool1'),
    Dropout(0.2),
    Conv2D(16, (3,3), padding='same', activation='relu', name='conv2'),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    Flatten(),
    Dense(128),
    Dropout(0.5),
    Dense(10, activation='softmax')])
model.compile(loss='categorical_crossentropy', \
              optimizer=Adam(0.001), \
              metrics=['accuracy'])

# Train the model with labeled sample
model.fit(x=xla, y=yla, \
          batch_size=20, epochs=30, \
          validation_data=(xte, yte), \
          shuffle=True)

# Train the model with labeled samples and unlabled samples
threshold = 0.9
for i in range(10):
    for j in range(0, len(xun), 10000):
        xbatch = xun[j:j+10000] # 无标签数据分 N 拨迭代（这一步似乎很重要）
        pbatch = model.predict(xbatch)
        idx = np.max(pbatch, axis=-1)
        xbatch = xbatch[idx>threshold]
        pbatch = pbatch[idx>threshold]
        pbatch = np.argmax(pbatch, axis=-1)
        pbatch = np.eye(10)[pbatch]
        print('SSL iteratin {}-{}, threshold: {}, {} samples selected'.format(i,j, threshold, xbatch.shape[0]))

        model.fit(xbatch, pbatch, \
              batch_size=128, epochs=1, \
              validation_data = (xte, yte), \
              shuffle = True)
    threshold -= 0.1

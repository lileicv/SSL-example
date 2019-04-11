import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential

from dataset import gaussian4
from vis import plot_decision_boundary

def Net():
    model = Sequential([
        Dense(5, input_shape=[2], activation='relu'),
        Dense(4, activation='softmax')
    ])
    return model


# dataset
x, y, un = gaussian4()

# Build Model
model = Net()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train SL
print('Train with labeled samples')
model.fit(x,y,batch_size=4, epochs=1000, verbose=0)
plot_decision_boundary(model, x, un, './images/boundary-st-sl.png')

# Train SSL
for i in range(10):
    print('Train with unlabeled samples, ', i)
    yun = model.predict(un)
    yun = np.argmax(yun, axis=-1)
    yun = np.eye(4)[yun]
    model.fit(un, yun, epochs=10)
    plot_decision_boundary(model, x, un, './images/boundary-st-ssl-{}.png'.format(i))


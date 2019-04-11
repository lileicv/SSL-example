'''
Visualization
'''
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

def plot_decision_boundary(model, x, un, save_path='./decision_boundary.png'):
    '''
    Visualize the decision boundary
    '''
    d = np.concatenate([x,un], axis=0)
    x_min = d[:,0].min()-0.5
    x_max = d[:,0].max()+0.5
    y_min = d[:,0].min()-0.5
    y_max = d[:,0].max()+0.5
    h = 0.05
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=-1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(un[:, 0], un[:, 1], c='k')
    plt.scatter(x[:,0], x[:,1], c='g')
    if save_path is not None:
        plt.savefig(save_path)



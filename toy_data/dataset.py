'''
Build the toy dataset
'''

import numpy as np

def gaussian4(one_hot=True):
    ''' Make toy dataset
        4 sets of sample points that match the gaussian distribution
        4 sample points with labels
    '''
    # unlabeled
    un = []
    std = [[0.1,0],[0,0.1]]
    for mean in [[1,0],[0,1],[-1,0],[0,-1]]:
        samples = np.random.multivariate_normal(mean, std, 200)
        un.append(samples)
    un = np.concatenate(un, axis=0).astype(np.float32)
    # labeled
    x = np.array([[1,-0.6],[0.6,1],[-1,0.6],[-0.6,-1]], np.float32)
    y = np.array([0,1,2,3], np.int)
    if one_hot:
        y = np.eye(4)[y]
    return x, y, un


if __name__=='__main__':
    x,y,un = gaussian4()
    print(x)
    print(y)
    print(un.shape)


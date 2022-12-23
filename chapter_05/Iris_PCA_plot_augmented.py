import numpy as np
import matplotlib.pylab as plt
from sklearn import decomposition

# load the data and exclude class 0 from so we only keep 2 dimensions
x = np.load("../data/iris/iris_features.npy")[:,:2]
y = np.load("../data/iris/iris_labels.npy")
a = np.load("../data/iris/iris_train_features_augmented.npy")[:,:2]
b = np.load("../data/iris/iris_train_labels_augmented.npy")
idx = np.where(y != 0)
x = x[idx]
idx = np.where(b != 0)
a = a[idx]

print(x)
print(a)

# subtract the per feature means to center the center the data about point (0,0)
# x[:,0] -= x[:,0].mean()
# x[:,1] -= x[:,1].mean()
# a[:,0] -= a[:,0].mean()
# a[:,1] -= a[:,1].mean()


# Create the PCA object and fit the the iris data
pca_x = decomposition.PCA(n_components=2)
pca_x.fit(x)
v_x = pca_x.explained_variance_ratio_
pca_a = decomposition.PCA(n_components=2)
pca_a.fit(a)
v_a = pca_a.explained_variance_ratio_

# create the plot
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(x[:,0],x[:,1],marker='o',color='b')
ax1.scatter(a[:,0],a[:,1],marker='+',color='r')
ax1 = plt.axes()
x0 = v_x[0]*pca_x.components_[0,0]
y0 = v_x[0]*pca_x.components_[0,1]
ax1.arrow(0, 0, x0, y0, head_width=0.05, head_length=0.1, fc='b', ec='b')
x1 = v_x[1]*pca_x.components_[1,0]
y1 = v_x[1]*pca_x.components_[1,1]
ax1.arrow(0, 0, x1, y1, head_width=0.05, head_length=0.1, fc='b', ec='b')
a0 = v_a[0]*pca_a.components_[0,0]
b0 = v_a[0]*pca_a.components_[0,1]
ax1.arrow(0, 0, a0, b0, head_width=0.05, head_length=0.1, fc='r', ec='r')
a1 = v_a[1]*pca_a.components_[1,0]
b1 = v_a[1]*pca_a.components_[1,1]
ax1.arrow(0, 0, a1, b1, head_width=0.05, head_length=0.1, fc='r', ec='r')
plt.xlabel("$x_0$", fontsize=16)
plt.ylabel("$x_1$", fontsize=16)
plt.show()
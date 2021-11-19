from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

im = Image.open('flower1.jpg')
plt.imshow(im)
newdata = np.array(im).reshape((-1, 3))
shape = newdata.shape
print(shape)
gmm = GaussianMixture(n_components=2, covariance_type="tied")
#高斯分布主要是依据颜色的分布进行划分，n_components越小，划分的越粗糙
gmm = gmm.fit(newdata)

cluster = gmm.predict(newdata)
# cluster = gmm.predict_proba(newdata)
# cluster = (cluster > 0.98).argmax(axis=1)
cluster = cluster.reshape(842, 474)
print(cluster.shape)
plt.imshow(cluster)
plt.show()


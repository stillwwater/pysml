# Simple ML

Simple implementations of some ML algorithms in Python.

## Usage

```python
>>> import pysml as sml
>>> import numpy as np

>>> X = np.array([[ 1., 2.], [ 1., 4.], [ 1., 0.],
...               [10., 2.], [10., 4.], [10., 0.]])

>>> kmeans = sml.kmeans(X, n_clusters=2, random_state=0)
>>> kmeans.labels
array([0, 0, 0, 1, 1, 1], dtype=int32)            

>>> kmeans.predict([[0., 0.], [12., 3.]])
array([0, 1], dtype=int32)

>>> kmeans.centers
array([[ 1.,  2.],
       [10.,  2.]])
```




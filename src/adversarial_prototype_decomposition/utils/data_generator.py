import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

n=100
x1=(np.random.randn(n,2) * [2,1],
    np.random.randn(n,2) * [1,2] + [0,3],
    np.random.randn(n,2) * [1, 0.7] + [3,3],
    np.random.randn(n,2) * [1, 0.7] + [-3,3],
    np.random.randn(n,2) * [0.8, 0.7] + [0,-3],
    )

y1=(
     np.ones((n,)),
     np.ones((n,)),
     -np.ones((n,)),
     -np.ones((n,)),
     -np.ones((n,))
    )

x= np.vstack(x1)
y= np.vstack(y1)

dat = np.hstack((x,np.reshape(y,(-1,1))))
plt.figure(1)
plt.clf()
plt.scatter(x[:,0],x[:,1],c=y)
df = pd.DataFrame(dat,columns=["a1","a2","Class"])

df.to_csv("Data/4_clust.csv",index=False)
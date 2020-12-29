import autograd.numpy as np
import matplotlib.pyplot as plt

dx = lambda x,y: x*(3-x-2*y)
dy = lambda x,y: y*(2-x-y)

lotka_volterra = lambda x, y: np.vstack((x*(3-x-2*y), y*(2-x-y)))
df1dx = lambda x, y: 3-2*x-2*y
df1dy = lambda x: -2*x
df2dx = -1
df2dy = lambda x,y: 1-x-y

x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)

x, y = np.meshgrid(x, y)

x = x.ravel()
y = y.ravel()

xx = dx(x,y)
yy = dy(x,y)
#plt.quiver(x,y, xx, )
#plt.show()

df1x = df1dx(x,y)
df1y = df1dy(x)
df2x = np.repeat(df2dx, len(x))
df2y = df2dy(x,y)

jacobians = np.array([[df1x, df1y], [df2x, df2y]])
jacobians = np.reshape(jacobians, (len(x), 2,2))
evals, _ = np.linalg.eig(jacobians)

plt.hist(evals, bins=1000)
plt.show()



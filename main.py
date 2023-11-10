import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 3, 0.05)
plt.grid(visible=True)
# plt.plot(x, x**2)
# plt.plot(x, 2*x)
# y=m*(x-x0)+y0  => y - y0 = m (x - x0)
# plt.plot(x, 2*(x-1)+1 )
plt.plot(x, np.cosh(x) )
plt.show()
# Is the data distance really convex? Answer: No.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# grid space.
x = np.linspace(-2, 2, 10000)

# set up two quadratics.
y1 = (x - 1) ** 2.0
y2 = (x + 1) ** 2.0

z1 = np.stack((y1, y2))

print(z1.shape)

z_min = np.min(z1, axis=0)
z_softmin = -logsumexp(-z1, axis=0)

plt.figure()
plt.plot(x, z_min)
plt.plot(x, z_softmin)
plt.show()

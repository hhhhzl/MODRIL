import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Piece-wise  “Cliff” (1-D)
# --------------------------------------------------
x = np.linspace(-3, 3, 2000)


def r_cliff(x):
    return np.where(np.abs(x) < .2, 1 - np.abs(x),
                    np.where(np.abs(x) < 2, -0.8, 0.))


y = r_cliff(x)
plt.figure(figsize=(4, 2))
plt.plot(x, y)
# plt.show()

scale = 20
s = np.repeat(x, scale)
a = np.repeat(y, scale)
noise = np.random.normal(0, 0.2, a.shape)
a_noise = a + noise
plt.scatter(s, a_noise, s=3, alpha=0.02)
plt.show()

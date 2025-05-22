import numpy as np
import matplotlib.pyplot as plt


def sine(x):
    return np.sin(2 * np.pi * 1 * x)


def r_cliff(x):
    # Piece-wise  “Cliff” (1-D)
    return np.where(np.abs(x) < .2, 1 - np.abs(x), np.where(np.abs(x) < 2, -0.8, 0.))


def multi_basin(x):
    # Multi-Basin  “Himmelblau-1D slice”
    return -((x ** 2 - 11) ** 2 + (x - 7) ** 2) + 200


def rastrigin_patch(x, y):
    return - (20 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y)))


def sparse_gate(x, y):
    return np.where((np.abs(y) < .2) & (x > 0), 1, np.where((np.abs(x) < .2) & (y > 0), 1, 0))


def draw(f_name='rastrigin', w_limit=[-3, 3], sample=2000):
    funcs = {
        'sine':sine,
        'cliff':r_cliff,
        'basin':multi_basin,
        'rastrigin':rastrigin_patch,
        'gate': sparse_gate
    }
    if f_name in ['sine', 'cliff', 'basin']:
        x = np.linspace(w_limit[0], w_limit[1], sample)
        y = funcs[f_name](x)
        plt.figure(figsize=(4, 2))
        plt.plot(x, y)
        plt.show()
        scale = 20
        s = np.repeat(x, scale)
        a = np.repeat(y, scale)
        noise = np.random.normal(0, 0.2, a.shape)
        a_noise = a + noise
        plt.scatter(s, a_noise, s=3, alpha=0.02)
        plt.title(f'{f_name} reward')
        plt.show()
    else:
        grid = np.linspace(w_limit[0], w_limit[1], sample)
        x, y = np.meshgrid(grid, grid)
        z = funcs[f_name](x, y)
        plt.figure(figsize=(4,3))
        plt.contourf(x, y, z, 40, cmap='coolwarm')
        plt.colorbar()
        # N = 5000
        xs = np.random.uniform(-3, 3, sample)
        ys = np.random.uniform(-3, 3, sample)
        zs_true = funcs[f_name](xs, ys)
        noise = np.random.normal(0, 0.2, size=sample)
        zs_noisy = zs_true + noise
        plt.scatter(xs, ys, c=zs_noisy, s=8, alpha=0.15, linewidths=0)
        # plt.colorbar(label='noisy reward')
        plt.title('Rastrigin + noisy samples')
        plt.title(f'{f_name} reward')
        plt.show()

draw()



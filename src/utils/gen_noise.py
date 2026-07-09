import numpy as np


def gen_noise(mean, cov, size=1000):
    noise = np.random.multivariate_normal(mean, cov, size)
    return noise


if __name__ == "__main__":
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    noise = gen_noise(mean, cov, 10)

    print(noise)

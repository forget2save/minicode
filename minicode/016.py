# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, A, B, D, H, Q, R, P, X):
        self.A = A
        self.B = B
        self.D = D
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.X = X

    def func(self, x, y):
        return x @ y @ x.T

    def filter(self, Z, U=None, inputOn=False):
        # 计算预测值
        if inputOn:
            self.X = self.A @ self.X + self.B @ U
        else:
            self.X = self.A @ self.X
        # 计算协方差
        self.P = self.func(self.A, self.P) + self.func(self.D, self.Q)
        # 计算kalman增益
        self.K = self.P @ self.H.T @ np.linalg.inv(self.func(self.H, self.P) + self.R)
        # 更新预测值
        self.X = self.X + self.K @ (Z - self.H @ self.X)
        # 更新协方差
        self.P = (np.identity(self.K.shape[0]) - self.K @ self.H) @ self.P


class UnscentedKalmanFilter:
    def __init__(self, A, Q, R, P, X, Z, a):
        self.A = A
        self.Q = Q
        self.R = R
        self.P = P
        self.X = X
        self.Z = Z
        self.alpha = a
        self.dim = self.X.shape[0]
        self.k = 3 - self.dim + 1
        self.gamma = self.alpha * self.alpha * (self.dim + self.k)
        self.weight_m = [np.array([1 - self.dim / self.gamma])]
        self.weight_m.extend(
            [np.array([0.5 / self.gamma]) for _ in range(2 * self.dim)]
        )
        self.weight_c = [
            np.array([1 - self.dim / self.gamma + 1 - self.alpha * self.alpha + 2])
        ]
        self.weight_c.extend(
            [np.array([0.5 / self.gamma]) for _ in range(2 * self.dim)]
        )

    def MSE(self, x):
        return x @ x.T

    def f(self, x):
        return self.A @ x

    def h(self, x):
        return -np.log10(np.abs(x))

    def selectSigmaPoints(self):
        self.sigma = [self.X]
        tmpP = np.sqrt(np.abs(self.gamma * self.P))
        self.sigma.extend(
            [self.X + tmpP[:, i].reshape(tmpP.shape[0], 1) for i in range(self.dim)]
        )
        self.sigma.extend(
            [self.X - tmpP[:, i].reshape(tmpP.shape[0], 1) for i in range(self.dim)]
        )

    def predict(self):
        self.X = np.zeros_like(self.X)
        for x, w in zip(self.sigma, self.weight_m):
            x = self.f(x)
            self.X += w * x
        self.P = self.Q.copy()
        self.sigmaz = []
        self.Z = np.zeros_like(self.Z)
        for x, w in zip(self.sigma, self.weight_c):
            self.P += w * self.MSE(x - self.X)
            self.sigmaz.append(self.h(x[0]))
        for z, w in zip(self.sigmaz, self.weight_m):
            self.Z += w * z

    def update(self, Z):
        Pzz = self.R.copy()
        for z, w in zip(self.sigmaz, self.weight_c):
            Pzz += w * self.MSE(z - self.Z)
        Pxz = np.zeros((self.dim, self.Z.shape[0]))
        for x, z, w in zip(self.sigma, self.sigmaz, self.weight_c):
            Pxz += w * (x - self.X) @ (z - self.Z).T
        Kt = Pxz @ np.linalg.inv(Pzz)
        self.X += Kt @ (Z - self.Z)
        self.P -= Kt @ Pzz @ Kt.T


def jss():
    t = 100
    real_pos = np.zeros(t)
    observe_pos = np.zeros(t)
    ukf_pos = np.zeros(t)
    A = np.array([[1, 1], [0, 1]])
    Q = 0.5 ** 2 * np.array([[0.25, 0.5], [0.5, 1]])
    R = np.array([[0.1 ** 2]])
    P = np.array([[0.25, 0.5], [0.5, 1]])
    Z = np.array([[-2]], dtype=np.float)
    X = np.array([[1000, 20]], dtype=np.float).T
    UKF = UnscentedKalmanFilter(A, Q, R, P, X, Z, 1e-3)
    W = 0.5 * np.random.randn(t + 1)
    V = 0.1 * np.random.randn(t + 1)
    for i in range(1, t + 1):
        X[0] += 0.5 * W[i] + X[1]
        X[1] += W[i]
        real_pos[i - 1] = X[0]
        z = UKF.h(X[0]) + V[i]
        observe_pos[i - 1] = 10 ** (-z)
        UKF.selectSigmaPoints()
        UKF.predict()
        UKF.update(z)
        ukf_pos[i - 1] = UKF.X[0]
    obmse = np.sqrt(np.linalg.norm(observe_pos - real_pos))
    kfmse = np.sqrt(np.linalg.norm(ukf_pos - real_pos))
    return obmse, kfmse
    # plt.plot(real_pos, label="real")
    # plt.plot(observe_pos, label="observe")
    # plt.plot(ukf_pos, label="ukf")
    # plt.legend()
    # plt.show()


def main():
    A = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    B = np.zeros_like(A)
    D = np.array([[0.5, 0], [1, 0], [0, 0.5], [0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    Q = 5 * np.identity(2)
    R = 100 * np.identity(2)
    P = 1000 * np.identity(4)
    X = np.array([[25, -120, 10, 0]]).T
    model = KalmanFilter(A, B, D, H, Q, R, P, X)
    Z = np.array([[30, 11]]).T
    model.filter(Z)


if __name__ == "__main__":
    obs_err = np.zeros(100)
    ukf_err = np.zeros(100)
    for k in range(100):
        obs_err[k], ukf_err[k] = jss()
    plt.plot(obs_err, label="obs")
    plt.plot(ukf_err, label="ukf")
    plt.legend()
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.xlabel("t")
    plt.ylabel("rmse")
    plt.savefig("if.jpg")
    print(np.mean(obs_err), np.mean(ukf_err))

# coding:utf-8
import numpy as np


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
    main()

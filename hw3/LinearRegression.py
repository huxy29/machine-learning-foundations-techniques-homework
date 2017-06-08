import numpy as np

class LinearRegression():

    def sign(self, x):
        if x<=0:
            return -1
        return 1

    def gen_data(self, N=1000):
        data = np.ones((N, 3))
        target = np.zeros(N)
        for i in range(N):
            x1 = np.random.random()*2 - 1
            x2 = np.random.random()*2 - 1
            y = self.sign(x1**2 + x2**2 -0.6)
            # add noise
            if np.random.random()<0.1:
                y = -y
            data[i, 1:] = [x1, x2]
            target[i] = y
        return data, target

    def feat_transform(self, X):
        N = X.shape[0]
        Z = np.ones((N, 6))
        Z[:, 1] = X[:, 1]
        Z[:, 2] = X[:, 2]
        Z[:, 3] = X[:, 1]*X[:, 2]
        Z[:, 4] = X[:, 1]**2
        Z[:, 5] = X[:, 2]**2
        return Z

    def train(self, X, Y):
        # numpy.linalg.pinv calculate pesudo inverse
        pinv = np.linalg.pinv(X)
        w = np.dot(pinv, Y)
        self.w = w

    def calc_err_rate(self, X, Y):
        a = Y*np.dot(X, self.w)
        err_cnt = 0
        N = X.shape[0]
        for i in a:
            if i<=0:
                err_cnt += 1
        return err_cnt/N

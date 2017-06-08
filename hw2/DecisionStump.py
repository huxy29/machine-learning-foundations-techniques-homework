import numpy as np

class DecisionStump():

    def gen_data(self, size=20):
        X = np.random.uniform(-1, 1, size=size)
        y = np.ones(size)
        for i in range(size):
            if X[i]<0:
                y[i]=-1
            if np.random.random_sample()<0.2:
                y[i]=-y[i]
        return X, y

    def calc_err_rate(self, X, y, s, theta):
        m = X.size
        err_cnt = 0
        for i in range(m):
            if s*(X[i]-theta)*y[i]<0:
                err_cnt+=1
        return err_cnt/m

    def train_1d(self, X, y):
        sorted_X = np.sort(X)
        theta_candidates = []
        for i in range(sorted_X.size-1):
            theta_candidates.append((sorted_X[i]+sorted_X[i+1])/2)
        s_candidates = [-1, 1]
        opt_s = 1
        opt_theta = 0
        opt_err = self.calc_err_rate(X, y, opt_s, opt_theta)
        for s in s_candidates:
            for theta in theta_candidates:
                err = self.calc_err_rate(X, y, s, theta)
                if err<opt_err:
                    opt_err = err
                    opt_s = s
                    opt_theta = theta
        return opt_err, opt_s, opt_theta

    def calc_Eout(self, s, theta, noise=0.2):
        mu = (1-s+s*np.fabs(theta))/2
        Eout = (1-noise)*mu + noise*(1-mu)
        return Eout

    def load_data(self, fp):
        with open(fp, 'r') as f:
            lines = f.readlines()
        m = len(lines)
        row0 = lines[0].strip().split(' ')
        d = len(row0)-1
        X = np.ones((m, d))
        y = np.zeros(m)
        i = 0
        for line in lines:
            row = line.strip().split(' ')
            X[i, :] = row[:-1]
            y[i] = row[-1]
            i += 1
        return X, y

    def train(self, X, y):
        opt_i = 0
        opt_s = 1
        opt_theta = 0
        opt_err = 1
        m, d = X.shape
        for i in range(d):
            X_i = X[:, i]
            err, s, theta = self.train_1d(X_i, y)
            if err<opt_err:
                opt_i = i
                opt_s = s
                opt_theta = theta
                opt_err = err
        self.opt_err = opt_err
        self.opt_i = opt_i
        self.opt_s = opt_s
        self.opt_theta = opt_theta

    def verify(self, X, y):
        X_i = X[:, self.opt_i]
        m = X.shape[0]
        err_cnt = 0
        for j in range(m):
            if self.opt_s*(X_i[j]-self.opt_theta)*y[j]<0:
                err_cnt += 1
        return err_cnt/m

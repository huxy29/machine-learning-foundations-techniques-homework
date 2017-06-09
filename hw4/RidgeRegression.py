import numpy as np

class RidgeRegression():

    def load_data(self, fp):
        with open(fp, 'r') as f:
            lines = f.readlines()
        N = len(lines)
        d = len(lines[0].strip().split(' ')) - 1
        X = np.ones((N, d+1))
        y = np.zeros(N)
        for i in range(N):
            row = lines[i].strip().split()
            X[i, 1:] = row[:-1]
            y[i] = row[-1]
        return X, y

    def train(self, X, y, lamda=11.26):
        d = X.shape[1]
        I = np.identity(d)
        A = np.linalg.inv(lamda*I + np.dot(np.transpose(X), X))
        self.w = np.dot(np.dot(A, np.transpose(X)), y)

    def calc_err_rate(self, X, y):
        a = y*np.dot(X, self.w)
        err_cnt = 0
        N = X.shape[0]
        for i in a:
            if i<=0:
                err_cnt += 1
        return err_cnt/N

    def find_minErr(self, list_of_Err):
        minIndex = 0
        minEin = 1
        i = 0
        for e in list_of_Err:
            if e<minEin:
                minEin = e
                minIndex = i
            i+=1
        return minIndex

    def cross_val_err(self, X, y, k=5, lamda=1):
        N = len(X)
        err_rate = 0
        for i in range(k):
            low = int(i*N/k)
            high = int((i+1)*N/k)
            Dval_X = X[low:high]
            Dval_y = y[low:high]
            Dtrain_X = np.row_stack((X[:low], X[high:]))
            Dtrain_y = np.append(y[:low], y[high:])
            self.train(Dtrain_X, Dtrain_y, lamda=lamda)
            err_rate += self.calc_err_rate(Dval_X, Dval_y)
        return err_rate/k

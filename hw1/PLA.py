import numpy as np

class PLA():

    def load_data(self, fp):
        with open(fp, 'r') as f:
            lines = f.readlines()
        m = len(lines)
        line0 = lines[0].split('\t')[0].split(' ')
        d = len(line0)
        X = np.ones((m, d+1))
        y = np.zeros(m)
        i = 0
        for line in lines:
            line = line.strip()
            X[i, 1:] = line.split('\t')[0].split(' ')
            y[i] = line.split('\t')[1]
            i += 1
        return X, y

    def sign(self, x):
        if x<=0:
            return -1
        return 1

    def train(self, X, y, random=False, eta=1):
        m, d = X.shape
        w = np.zeros(d)
        num_of_corrects = 0    # when it is equal to m, all examples are corecctly classified, then halts
        num_of_updates = 0
        index_of_last_mistake = 0
        index = np.arange(m)
        if random:
            np.random.shuffle(index)    # random visiting
        i = 0
        is_finished = False
        while not is_finished:
            if y[index[i]]==self.sign(np.dot(w, X[index[i]])):
                num_of_corrects += 1
            else:
                # update w(t+1)=w(t)+y*x
                w += eta * y[index[i]] * X[index[i]]
                num_of_updates += 1
                num_of_corrects = 0
                index_of_last_mistake = index[i]
            if i==m-1:
                i=0
            else:
                i+=1
            if num_of_corrects==m:
                is_finished = True
        self.w = w
        self.num_of_updates = num_of_updates
        self.index_of_last_mistake = index_of_last_mistake

    def calc_err_rate(self, w, X, y):
        err_cnt = 0
        m = len(X)
        for i in range(m):
            if y[i]!=self.sign(np.dot(w, X[i])):
                err_cnt += 1
        return err_cnt/m

    def train_pocket(self, X, y, iteration=50):
        m, d = X.shape
        w_pocket = np.zeros(d)
        pocket_err_rate = self.calc_err_rate(w_pocket, X, y)
        w = np.zeros(d)
        it = 0
        while it<iteration:
            index = np.random.randint(0, m)
            if y[index]!=self.sign(np.dot(w, X[index])):
                # update w(t+1)=w(t)+y*x
                w += y[index] * X[index]
                w_err_rate = self.calc_err_rate(w, X, y)
                if w_err_rate < pocket_err_rate:
                    w_pocket = w.copy()
                    pocket_err_rate = self.calc_err_rate(w_pocket, X, y)
                it += 1
        self.w_pocket = w_pocket
        self.w = w

    def verify_pocket(self, test_X, test_Y, pocket=True):
        if pocket:
            return self.calc_err_rate(self.w_pocket, test_X, test_Y)
        else:
            return self.calc_err_rate(self.w, test_X, test_Y)

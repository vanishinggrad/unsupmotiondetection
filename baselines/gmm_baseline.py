import pickle
import numpy as np
import scipy as sp
from sklearn.mixture import GaussianMixture

import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from dcn import aux



def do_gmm(x_train, y_train, x_test, y_test, num_reps = 30):

    test_accs = []
    for rep in range(num_reps):
        # print(f"{rep} of {num_reps}")
    
        gmm = GaussianMixture(n_components=2).fit(x_train)
        
        pred = gmm.predict(x_train)
        # print("training:")
        # print((pred == y_train).sum() / len(pred))
        # print((pred != y_train).sum() / len(pred))
        
        pred = gmm.predict(x_test)
        # print("testing:")
        # print((pred == y_test).sum() / len(pred))
        # print((pred != y_test).sum() / len(pred))

        acc = (pred == y_test).sum() / len(pred)
        acc = max(acc, 1 - acc)
        test_accs.append(acc)

    return test_accs

if __name__ == "__main__":

    ### lab
    
    prefix = 'data/lab/'
    
    f = pickle.loads(open(prefix+'train_data.pickle', 'rb').read())
    x_train, y_train = f['x'], f['y'].astype(np.int8)
    
    f2 = pickle.loads(open(prefix+'test_data.pickle', 'rb').read())
    x_test = np.concatenate([f2[k]['x'] for k in sorted(f2.keys())], axis=0)
    y_test = np.concatenate([f2[k]['y'] for k in sorted(f2.keys())], axis=0).astype(np.int8)

    print("lab:")
    lab = do_gmm(x_train, y_train, x_test, y_test)
    dcn = np.load('results/lab/logs_lab.npy')[:,0]
    pval = sp.stats.ttest_ind(lab, dcn, equal_var=False).pvalue
    print(np.mean(lab), np.std(lab), pval)

    ### house
    
    prefix = 'data/house/preprocessed/'
    x_train, y_train, x_test, y_test =  aux.get_train_test_data(prefix, "", np.float64, np.int8, (24,))
    y_train[y_train == 2] = 1
    y_test[y_test == 2] = 1
    
    print("house:")
    house = do_gmm(x_train, y_train, x_test, y_test)
    dcn = np.load('results/house/logs_house.npy')[:,0]
    pval = sp.stats.ttest_ind(house, dcn, equal_var=False).pvalue    
    print(np.mean(house), np.std(house), pval)
    



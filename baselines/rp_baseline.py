import pickle
import numpy as np
import scipy as sp
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as pt
import matplotlib as mpl

import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from dcn import aux



def do_rp(x_train, y_train, x_test, y_test, num_reps = 10):

    test_accs = {}
    for dim in range(2, x_train.shape[1]+1):
        test_accs[dim] = []
        for rep in range(num_reps):
            # print(f"{rep} of {num_reps}")
    
            A = np.random.randn(x_train.shape[1], dim)
            Q = sp.linalg.orth(A)
            # print(Q.shape)
        
            gmm = GaussianMixture(n_components=2).fit(x_train @ Q)
            
            # pred = gmm.predict(x_train)
            # print("training:")
            # print((pred == y_train).sum() / len(pred))
            # print((pred != y_train).sum() / len(pred))
            
            pred = gmm.predict(x_test @ Q)
            # print("testing:")
            # print((pred == y_test).sum() / len(pred))
            # print((pred != y_test).sum() / len(pred))
    
            acc = (pred == y_test).sum() / len(pred)
            acc = max(acc, 1 - acc)
            test_accs[dim].append(acc)

        print(f"dim {dim}: avg, std", np.mean(test_accs[dim]), np.std(test_accs[dim]))

    return test_accs

if __name__ == "__main__":

    gen_results = True

    if gen_results:

        ### lab
        
        prefix = 'data/lab/'
        
        f = pickle.loads(open(prefix+'train_data.pickle', 'rb').read())
        x_train, y_train = f['x'], f['y'].astype(np.int8)
        
        f2 = pickle.loads(open(prefix+'test_data.pickle', 'rb').read())
        x_test = np.concatenate([f2[k]['x'] for k in sorted(f2.keys())], axis=0)
        y_test = np.concatenate([f2[k]['y'] for k in sorted(f2.keys())], axis=0).astype(np.int8)
    
        print("lab:")
        lab = do_rp(x_train, y_train, x_test, y_test)
    
        ### house
        
        prefix = 'data/house/preprocessed/'
        x_train, y_train, x_test, y_test =  aux.get_train_test_data(prefix, "", np.float64, np.int8, (24,))
        y_train[y_train == 2] = 1
        y_test[y_test == 2] = 1
        
        print("house:")
        house = do_rp(x_train, y_train, x_test, y_test)
    
        with open("baselines/rp_baseline.pkl", "wb") as f: pickle.dump((lab, house), f)

    # plot results

    with open("baselines/rp_baseline.pkl", "rb") as f: (lab, house) = pickle.load(f)
    dims = sorted(lab.keys())

    lab = np.array([lab[dim] for dim in dims])
    house = np.array([house[dim] for dim in dims])

    print(f"lab {lab.mean(axis=1).argmax()}:")
    dcn = np.load('results/lab/logs_lab.npy')[:,0]
    pval = sp.stats.ttest_ind(lab[lab.mean(axis=1).argmax()], dcn, equal_var=False).pvalue
    print(np.mean(lab), np.std(lab), pval)
    
    print(f"house {house.mean(axis=1).argmax()}:")
    dcn = np.load('results/house/logs_house.npy')[:,0]
    pval = sp.stats.ttest_ind(house[house.mean(axis=1).argmax()], dcn, equal_var=False).pvalue
    print(np.mean(house), np.std(house), pval)
    
    mpl.rcParams["font.family"] = "serif"
    pt.figure(figsize=(5,2.25))

    pt.subplot(1,2,1)
    means, stds = lab.mean(axis=1), lab.std(axis=1)
    for dim, res in zip(dims, lab): pt.plot([dim]*len(res), res, 'k.')
    pt.fill_between(dims, means-stds, means+stds, color='k', alpha=.25)
    pt.plot(dims, means, 'k-')
    pt.plot([min(dims), max(dims)], [.9909]*2, 'k:')
    # pt.plot([min(dims), max(dims)], [.9909 - .0037]*2, 'k--')
    pt.xlabel("Projection dimension")
    pt.ylabel("Test accuracy")
    pt.ylim([.9, 1.0])
    pt.title("Lab")

    pt.subplot(1,2,2)
    means, stds = house.mean(axis=1), house.std(axis=1)
    for dim, res in zip(dims, house): pt.plot([dim]*len(res), res, 'k.')
    pt.fill_between(dims, means-stds, means+stds, color='k', alpha=.25)
    pt.plot(dims, means, 'k-')
    pt.plot([min(dims), max(dims)], [.9884]*2, 'k:')
    # pt.plot([min(dims), max(dims)], [.9884 - .0007]*2, 'k--')
    pt.xlabel("Projection dimension")
    pt.ylim([.95, 1.0])
    pt.title("House")

    pt.tight_layout()
    pt.savefig("baselines/rp_baseline.pdf")
    pt.show()

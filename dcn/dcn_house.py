
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from dcn import dcn_main, dcn_config, aux


prefix = 'data/house/preprocessed/'
x_train, y_train, x_test, y_test =  aux.get_train_test_data(prefix, "", np.float64, np.int8, (24,))
y_train[y_train == 2] = 1
y_test[y_test == 2] = 1




logs = [] # acc, class0_acc, class1_acc,nmi, ari

for i in range(dcn_config.num_total_runs):
    print(f" --- Run {i}--- ")
    x_train, y_train = shuffle(x_train, y_train)

    dcn = dcn_main.DCN(shapes=[x_train.shape[-1], *dcn_config.hidden_layers], n_clusters=dcn_config.n_clusters, lambd=dcn_config.lambd)

    dcn.pretrain(x=x_train, batch_size=dcn_config.batch_size, epochs=dcn_config.epochs_pretrain)

    dcn.compile_for_joint_training()
    dcn.init_centers(x_train)

    dcn.fit(x_train, y_train, batch_size=dcn_config.batch_size, epochs=dcn_config.epochs_train)

    best_mapping, _ = aux.clust_best_mapping(y_test, dcn.predict(x_test))
    class0_acc = accuracy_score(y_test[y_test == 0], best_mapping[y_test == 0])
    class1_acc = accuracy_score(y_test[y_test == 1], best_mapping[y_test == 1])

    final = dcn.metric(y_test, dcn.predict(x_test))
    logs.append((class0_acc, class1_acc, *final))
    
np.save("results/house/logs_house.npy", np.array(logs))
dcn.autoencoder.save_weights('dcn/saved_wts_house.h5')

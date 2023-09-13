from dcn import *
import aux
import dcn_config
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import numpy as np

import os
print(os.getcwd())
prefix = '../data/lab/'

f = pickle.loads(open(prefix+'train_data.pickle', 'rb').read())
x_train, y_train = f['x'], f['y'].astype(np.int8)

f2 = pickle.loads(open(prefix+'test_data.pickle', 'rb').read())


x_test = [f2[k]['x'] for k in sorted(f2.keys())]
y_test = [f2[k]['y'] for k in sorted(f2.keys())]
x_test_stacked = np.concatenate([f2[k]['x'] for k in sorted(f2.keys())], axis=0)
y_test_stacked = np.concatenate([f2[k]['y'] for k in sorted(f2.keys())], axis=0, dtype=np.int8)


def get_stats_over_runs():
    logs = []
    for i in range(1):
        print(f" --- Run {i}--- ")
        logs.append([])
        for ix, data in enumerate(zip(x_test, y_test)):
            best_mapping, accuracy = aux.clust_best_mapping(data[1], dcn.predict(data[0]))

            logs[-1].append([(accuracy_score(data[1][data[1] == 1], best_mapping[data[1] == 1])),
                             (accuracy_score(data[1][data[1] == 0], best_mapping[data[1] == 0])),
                             *dcn.metric(data[1], dcn.predict(data[0])),])

    return np.array(logs)


dcn = DCN(shapes=[x_train.shape[-1], *dcn_config.hidden_layers], n_clusters=dcn_config.n_clusters,
                  lambd=dcn_config.lambd)
dcn.pretrain(x=x_train, batch_size=dcn_config.batch_size, epochs=1)  #1 just to reuse the same code, weights will be over-written anyway
dcn.autoencoder.load_weights("saved_wts_house.h5")
dcn.compile_for_joint_training()
dcn.init_centers(x_train)


before = get_stats_over_runs()
before = np.squeeze(before.mean(0))[:, :2]

dcn.fit(x_train, y_train, batch_size=30, epochs=10)
after = get_stats_over_runs()
after = np.squeeze(after.mean(0))[:, :2]

fig, ax = plt.subplots(
                       dpi=100, sharex=True)
ind = np.arange(before.shape[0])
width = 0.3
rects1 = ax.bar(ind, before[:, 0], color='b' , width=width, label = "Human-Presence")
rects2 = ax.bar(ind, after[:, 0]-before[:, 0], bottom = before[:, 0], color = 'b', edgecolor = 'k',  width=width, hatch= '.', label = "")
rects3 = ax.bar(width+ind, before[:, 1], color = 'g', width=width, label = "Human-Free")
rects2 = ax.bar(width+ind, after[:, 1]-before[:, 1], bottom = before[:, 1], color = 'g', edgecolor = 'k',  width=width, hatch= '.', label = "")


ax.set_ylabel('Accuracy')
# ax.set_title('Motion Dee')
ax.grid(axis='y', linewidth=0.4)
ax.set_axisbelow(True)
ax.set_xticks(ind + width / 2, [f"d{d}" for d in 1+np.arange(before.shape[0])])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

plt.tight_layout()
plt.show()
fig.savefig("transfer_learning.pdf", dpi=100, transparent=True)

print(before.mean(axis=0), before.mean())
print(after.mean(axis=0), after.mean())

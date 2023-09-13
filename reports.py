import matplotlib.pyplot as plt
import numpy as np

print("For the house dataset")
logs = np.load('results/house/logs_house.npy')

print("mean (class0 acc, class1 acc, acc, nmi, ari)", (np.mean(logs, axis=0)*100).round(2))
print("std (class0 acc, class1 acc, acc, nmi, ari)", (np.std(logs, axis=0)*100).round(2))


# print("For the lab dataset")
# logs = np.load('results/lab/logs_lab.npy')
#
# print("mean (class0 acc, class1 acc, acc, nmi, ari)", (np.mean(logs, axis=0)*100).round(2))
# print("std (class0 acc, class1 acc, acc, nmi, ari)", (np.std(logs, axis=0)*100).round(2))

import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras import Input
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
import numpy as np
from scipy.spatial import distance as sd

import aux

tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()



SHOW_PROGRESS = True



def joint_loss(lambd, encoded_xs, centers_for_xs):
    def loss(y_true, y_pred):

        cost_clustering = tf.keras.losses.MeanSquaredError()(centers_for_xs, encoded_xs)
        cost_reconstruction = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        return cost_reconstruction + lambd*cost_clustering

    return loss


def get_autoencoder(shapes, act_method="relu", init_method="glorot_uniform"):


    n_hidden = len(shapes) - 1
    # input
    input_clustering = Input(shape=(shapes[-1],), name='clustering_input')

    x = Input(shape=(shapes[0],), name='input')
    h = x

    # hidden layers in encoder
    for i in range(n_hidden - 1):
        h = Dense(shapes[i + 1], activation=act_method, kernel_initializer=init_method, name='encoder_%d' % i)(h)

    # encoder-clustering layer
    h = Dense(shapes[-1], kernel_initializer=init_method, name='encoder_%d' % (n_hidden - 1))(
        h)

    y = h
    # hidden layers in decoder
    for i in range(n_hidden - 1, 0, -1):
        y = Dense(shapes[i], activation=act_method, kernel_initializer=init_method, name='decoder_%d' % i)(y)

    # output
    y = Dense(shapes[0], kernel_initializer=init_method, name='decoder_0')(y)

    ae_with_clustering = Model(inputs=[x, input_clustering], outputs=y, name="ae_clust")
    encoder = Model(inputs=x, outputs=h, name='encoder')

    return (ae_with_clustering,
            encoder,
            input_clustering)


class DCN(object):
    def __init__(self, shapes, n_clusters, lambd=0.5, init_method='glorot_uniform'):
        super(DCN, self).__init__()

        self.all_pred = None
        self.shapes = shapes
        self.input_dim = shapes[0]
        self.n_hidden = len(self.shapes) - 1

        self.n_clust = n_clusters
        self.lambd = lambd
        self.autoencoder, self.encoder, self.clustering_input = get_autoencoder(self.shapes, init_method=init_method)

        self.clust_centers = np.zeros((self.n_clust, self.shapes[-1]))
        self.clust_counts = 100 * np.ones(self.n_clust, dtype="int32")

    def pretrain(self, x, y=None, epochs=200, batch_size=256, save_dir="."):
        if SHOW_PROGRESS:
            print('...Pretraining...')

        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')

        initial_centers = np.zeros((x.shape[0], self.shapes[-1]), dtype='int32')
        self.autoencoder.fit([x, initial_centers], x, batch_size=batch_size, epochs=epochs)

        # self.autoencoder.save_weights(save_dir + '/pre_trained_weights.h5')
        # print('Weights for pre-training model have been saved to %s/pre_trained_weights.h5' % save_dir)
        self.pretrained = True

    def init_centers(self, x, y=None):

        # init self.centers
        kmeans = KMeans(n_clusters=self.n_clust)
        kmeans.fit(self.encoder.predict(x))
        self.all_pred = kmeans.labels_


        self.clust_centers = kmeans.cluster_centers_
        print('centers-', self.clust_centers)

        if y is not None:
            self.metric(y, self.all_pred)

    def compile_for_joint_training(self):
        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                                 loss=joint_loss(self.lambd, self.encoder.output, self.clustering_input))

    def fit(self, x, y, epochs, batch_size=256, save_dir='.'):

        m = x.shape[0]
        for epoch in range(epochs):
            cost = []  # all cost

            for batch_index in range(int(m // batch_size)):
                x_batch = x[batch_index * batch_size:(batch_index + 1) * batch_size, :]

                centers_for_xs = self.clust_centers[self.all_pred[batch_index * batch_size:(batch_index + 1) * batch_size]]

                c1 = self.autoencoder.train_on_batch([x_batch, centers_for_xs], x_batch)
                cost.append(c1)

                x_encoded = self.encoder.predict(x_batch)
                # update k-means
                self.all_pred[
                batch_index * batch_size:(batch_index + 1) * batch_size], self.clust_centers, self.clust_counts = self.batch_km(
                    x_encoded, self.clust_centers, self.clust_counts)

            if epoch % 10 == 0:
                acc, nmi, ari = self.metric(y, self.predict(x))
                print('epoch:', epoch, '|', 'clust counts', self.clust_counts, '|', ' cost:', np.mean(cost), '|', 'acc:', acc, '|', 'nmi:', nmi, '|', 'ari:', ari)


        # print('saving model to:', save_dir + '/DCN_model_final.h5')
        # self.autoencoder.save_weights(save_dir + '/saved_model_house.h5')

    def batch_km(self, data, center, count):
        """
        Function to perform a KMeans update on a batch of data, center is the
        centroid matrix from last iteration.
        """
        N = data.shape[0]
        K = center.shape[0]

        # update assignments
        idx = np.zeros(N, dtype='int32')
        for i in range(N):
            dist = np.inf
            ind = 0
            for j in range(K):
                temp_dist = np.linalg.norm(data[i] - center[j])
                if temp_dist < dist:
                    dist = temp_dist
                    ind = j
            idx[i] = ind

        # update centroids
        center_new = center
        for i in range(N):
            c = idx[i]
            count[c] += 1
            eta = 1.0 / count[c]
            center_new[c] = (1 - eta) * center_new[c] + eta * data[i]
        center_new.astype(np.float32)
        return idx, center_new, count

    def get_clust_labels_and_centers(self, encoded):

        distances = sd.cdist(encoded, self.clust_centers)
        cluster_labels = np.argmin(distances, axis=1)
        cluster_centers = self.clust_centers[cluster_labels]
        return cluster_centers, cluster_labels

    def load_weights(self, weights):
        self.autoencoder.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        x_encoded = self.encoder.predict(x)
        cluster_centers, cluster_labels = self.get_clust_labels_and_centers(x_encoded)
        return cluster_labels

    def metric(self, y, y_pred):
        acc = np.round(aux.acc(y, y_pred), 4)
        nmi = np.round(aux.nmi(y, y_pred), 4)
        ari = np.round(aux.ari(y, y_pred), 4)
        #print('acc:', acc, '|', 'nmi:', nmi, '|', 'ari:', ari)
        return (acc, nmi, ari)








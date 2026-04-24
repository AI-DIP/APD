import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons
from adversarial_prototype_decomposition.classifier.classifiers import APD_Multi_Classifier
from imblearn.under_sampling import ClusterCentroids
from collections import Counter

# Compute 3 prototypes per class using KMeans
def compute_prototypes(X, y, k=3):
    prototypes = []
    proto_labels = []
    for cls in np.unique(y):
        X_cls = X[y == cls]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cls)
        prototypes.append(kmeans.cluster_centers_)
        proto_labels.extend([cls] * k)
    return np.vstack(prototypes), np.array(proto_labels)

N_PROTO = 4

# Load and normalize data
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

proto_selection=ClusterCentroids(sampling_strategy={0: N_PROTO, 1: N_PROTO,2:N_PROTO})

apd = APD_Multi_Classifier(proto_selection,prune_regions=False, )
apd.fit(X_scaled,y)
# print(prototypes, proto_labels)
prototypes, proto_labels = apd.get_protos()
protos_id = np.array(sorted(set(sum(map(apd.proto_ensemble_.unpairCantor, apd.regions_), ()))))
protos_id_to_row = dict(zip(protos_id,range(len(protos_id))))
PX = apd.proto_ensemble_.proto[protos_id]
PY = apd.proto_ensemble_.proto_labels[protos_id]

# print(prototypes, proto_labels)

# Initial axes
x_idx, y_idx, z_idx = 0, 1, 2
is_3d = False

# Plot setup
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.3)

colors = ['red', 'green', 'blue']

# --- Dodajemy globalną siatkę do granic decyzji ---
decision_grid = None
decision_preds = None
grid_resolution = 100  # liczba punktów na osi

# --- GLOBALNE ---
decision_grid_all = None   # pełna siatka w wymiarach wszystkich cech
decision_preds_all = None  # predykcje dla pełnej siatki
n_points_per_feature = 10  # liczba punktów w każdej osi (zmniejszona dla wydajności)

# --- LICZENIE GRANIC DECYZJI RAZ, DLA WSZYSTKICH CECH ---
def compute_decision_boundary_all_features():
    global decision_grid_all, decision_preds_all
    n_features = X_scaled.shape[1]
    linspaces = []
    for i in range(n_features):
        f_min, f_max = X_scaled[:, i].min() - 0.5, X_scaled[:, i].max() + 0.5
        linspaces.append(np.linspace(f_min, f_max, n_points_per_feature))

    # Generujemy pełną siatkę w n-wymiarach
    mesh = np.meshgrid(*linspaces, indexing='ij')
    grid_points = np.stack([m.ravel() for m in mesh], axis=-1)  # (n_points^n_features, n_features)

    # Predykcje raz
    decision_preds_all = apd.predict(grid_points)
    decision_grid_all = mesh

def plot_decision_boundary():
    n_features = X_scaled.shape[1]

    if not is_3d:
        # ----------------- 2D -----------------
        idx_slices = [slice(None)] * n_features
        for i in range(n_features):
            if i != x_idx and i != y_idx:
                idx_slices[i] = 0

        Z = decision_preds_all.reshape([n_points_per_feature]*n_features)[tuple(idx_slices)]
        X_grid = decision_grid_all[x_idx][tuple(idx_slices)]
        Y_grid = decision_grid_all[y_idx][tuple(idx_slices)]
        X_grid = np.squeeze(X_grid)
        Y_grid = np.squeeze(Y_grid)
        Z = np.squeeze(Z)

        ax.contourf(X_grid, Y_grid, Z, alpha=0.2,
                    levels=np.arange(len(np.unique(y)) + 1) - 0.5,
                    colors=colors)
    else:
        # ----------------- 3D -----------------
        # Wybieramy trzy wymiary do wykresu
        idx_slices = [slice(None)] * n_features
        for i in range(n_features):
            if i not in (x_idx, y_idx, z_idx):
                idx_slices[i] = 0  # reszta wymiarów = pierwsza warstwa

        Z = decision_preds_all.reshape([n_points_per_feature]*n_features)[tuple(idx_slices)]
        X_grid = decision_grid_all[x_idx][tuple(idx_slices)]
        Y_grid = decision_grid_all[y_idx][tuple(idx_slices)]
        Z_grid = decision_grid_all[z_idx][tuple(idx_slices)]

        # Podpróbkowanie punktów dla wydajności
        step = max(1, n_points_per_feature // 10)
        ax.scatter(X_grid[::step, ::step, ::step].ravel(),
                   Y_grid[::step, ::step, ::step].ravel(),
                   Z_grid[::step, ::step, ::step].ravel(),
                   c=Z.ravel()[::step**3],
                   cmap=plt.cm.RdYlBu,
                   alpha=0.1,
                   marker='.',
                   s = 4000)

# --- Aktualizacja update_plot ---
def update_plot():
    global ax, decision_grid, decision_preds
    ax.clear()

    # Po zmianie osi/granicy, przeliczamy siatkę
    # compute_decision_boundary()

    # 3D
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        for cls in np.unique(y):
            ax.scatter(X_scaled[y == cls, x_idx],
                       X_scaled[y == cls, y_idx],
                       X_scaled[y == cls, z_idx],
                       label=iris.target_names[cls],
                       color=colors[cls], alpha=0.6)
            for pair in apd.regions_:
                i, j = apd.proto_ensemble_.unpairCantor(pair)
                _x = [PX[protos_id_to_row[i]][x_idx], PX[protos_id_to_row[j]][x_idx]]
                _y = [PX[protos_id_to_row[i]][y_idx], PX[protos_id_to_row[j]][y_idx]]
                _z = [PX[protos_id_to_row[i]][z_idx], PX[protos_id_to_row[j]][z_idx]]
                ax.plot(_x,_y,_z,'r')

        # prototypes
        for cls in np.unique(proto_labels):
            ax.scatter(prototypes[proto_labels == cls, x_idx],
                       prototypes[proto_labels == cls, y_idx],
                       prototypes[proto_labels == cls, z_idx],
                       color=colors[cls],
                       marker='X',
                       s=150,
                       edgecolor='black',
                       label=f'Prototypes {iris.target_names[cls]}')

        #plot_decision_boundary()
        ax.set_zlabel(feature_names[z_idx])

    # 2D
    else:
        ax = fig.add_subplot(111)
        for cls in np.unique(y):
            ax.scatter(X_scaled[y == cls, x_idx],
                       X_scaled[y == cls, y_idx],
                       label=iris.target_names[cls],
                       color=colors[cls], alpha=0.6)
            for pair in apd.regions_:
                i, j = apd.proto_ensemble_.unpairCantor(pair)
                _x = [PX[protos_id_to_row[i]][x_idx], PX[protos_id_to_row[j]][x_idx]]
                _y = [PX[protos_id_to_row[i]][y_idx], PX[protos_id_to_row[j]][y_idx]]
                ax.plot(_x,_y,'r')

        # prototypes
        for cls in np.unique(proto_labels):
            ax.scatter(prototypes[proto_labels == cls, x_idx],
                       prototypes[proto_labels == cls, y_idx],
                       color=colors[cls],
                       marker='X',
                       s=150,
                       edgecolor='black',
                       label=f'Prototypes {iris.target_names[cls]}')

        plot_decision_boundary()

    ax.set_xlabel(feature_names[x_idx])
    ax.set_ylabel(feature_names[y_idx])
    ax.legend()
    plt.draw()

compute_decision_boundary_all_features()

# UI elements
rax_x = plt.axes([0.05, 0.7, 0.2, 0.2])
rax_y = plt.axes([0.05, 0.45, 0.2, 0.2])
rax_z = plt.axes([0.05, 0.2, 0.2, 0.2])
rax_mode = plt.axes([0.05, 0.05, 0.2, 0.1])

radio_x = RadioButtons(rax_x, feature_names)
radio_y = RadioButtons(rax_y, feature_names)
radio_z = RadioButtons(rax_z, feature_names)
radio_mode = RadioButtons(rax_mode, ['2D', '3D'])

# Callbacks
def set_x(label):
    global x_idx
    x_idx = feature_names.index(label)
    update_plot()

def set_y(label):
    global y_idx
    y_idx = feature_names.index(label)
    update_plot()

def set_z(label):
    global z_idx
    z_idx = feature_names.index(label)
    update_plot()

def set_mode(label):
    global is_3d
    is_3d = (label == '3D')
    update_plot()

radio_x.on_clicked(set_x)
radio_y.on_clicked(set_y)
radio_z.on_clicked(set_z)
radio_mode.on_clicked(set_mode)

# Initial plot
update_plot()
plt.show()
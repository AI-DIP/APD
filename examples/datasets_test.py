import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from sklearn.datasets import (
    make_blobs,
    make_moons,
    make_circles,
    make_gaussian_quantiles,
    make_multilabel_classification
)
import tkinter as tk
from tkinter import ttk
from imblearn.under_sampling import ClusterCentroids
from adversarial_prototype_decomposition.classifier.classifiers import APD_Multi_Classifier, APD_Multi_Classifier_Proto_Train
from adversarial_prototype_decomposition.sampler.glvq_sampler import GLVQ_Sampler

from matplotlib.colors import ListedColormap

CLASS_COLORS = None

# --- Generowanie danych ---
def generate_data(dataset_type, n_samples, n_classes, noise):
    if dataset_type == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=n_classes, random_state=42)
    elif dataset_type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif dataset_type == "gaussian":
        X, y = make_gaussian_quantiles(n_samples=n_samples, n_classes=n_classes, random_state=42)
    elif dataset_type == "multilabel":
        X, Y = make_multilabel_classification(n_samples=n_samples, n_features=2, n_classes=n_classes, random_state=42)
        y = np.argmax(Y, axis=1)
    else:
        X, y = np.array([]), np.array([])
    return X, y

def get_class_colors(n_classes):
    cmap = plt.cm.get_cmap("tab10", n_classes)
    return [cmap(i) for i in range(n_classes)]

# --- Inicjalizacja parametrów ---
params = {
    "dataset": "blobs",
    "n_samples": 300,
    "n_classes": 3,
    "noise": 0.1,
    "min_support": 100,
    "unbalanced_rate": 0.1,
    "prune_regions": True,
    "n_proto": 3,
    "show_proto": True,
    "show_edges": True,
    "show_boundary": True
}

# --- Obiekt do przechowywania precomputed danych ---
precomputed = {
    "X": None,
    "y": None,
    "clf": None,
    "y_pred": None,
    "xx": None,
    "yy": None,
    "Z": None,
    "proto_acc": None
}

# --- Matplotlib ---
fig, ax = plt.subplots()
fig2, ax_proto = plt.subplots()
fig3, ax_regions = plt.subplots()
plt.subplots_adjust(left=0.3, right=0.95)


def prototype_assignment_accuracy(X, y, prototypes, proto_labels):
    # dla każdego punktu szukamy najbliższego prototypu
    acc = 0
    for i in range(len(X)):
        dists = np.linalg.norm(prototypes - X[i], axis=1)
        nearest_idx = np.argmin(dists)
        if proto_labels[nearest_idx] == y[i]:
            acc += 1
    return acc / len(X)
# --- Funkcja obliczająca APD i wyniki ---
def compute_apd():
    X, y = generate_data(params["dataset"], params["n_samples"], params["n_classes"], params["noise"])
    n_classes = len(np.unique(y))
    precomputed["class_colors"] = get_class_colors(n_classes)
    precomputed["X"], precomputed["y"] = X, y

    if X.size == 0:
        precomputed.update({"clf": None, "y_pred": None, "xx": None, "yy": None, "Z": None})
        return

    sampling_strategy = {cls: params["n_proto"] for cls in np.unique(y)}
    proto_selection = GLVQ_Sampler(prototype_n_per_class=params["n_proto"], random="kmeans")#ClusterCentroids(sampling_strategy=sampling_strategy)

    clf = APD_Multi_Classifier_Proto_Train(
        min_support=params["min_support"],
        unbalanced_rate=params["unbalanced_rate"],
        prune_regions=params["prune_regions"],
        proto_selection=proto_selection,
        minimum_regions=2,
        apd_type="apd_multi_class",
    )

    clf.fit(X, y)
    y_pred = clf.predict(X)
    prototypes, proto_labels = clf.get_protos()
    proto_acc = prototype_assignment_accuracy(X, y, prototypes, proto_labels)

    xx = yy = Z = R = None
    if params["show_boundary"]:
        h = 0.05
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        R = clf.proto_ensemble_.assign_regions(np.c_[xx.ravel(), yy.ravel()], list(clf.proto_ensemble_.pairs.keys()))

    precomputed.update({
        "clf": clf,
        "y_pred": y_pred,
        "xx": xx,
        "yy": yy,
        "Z": Z,
        "R": R,
        "proto_acc": proto_acc
    })

def predict_by_prototypes(X, prototypes, proto_labels):
    preds = []
    for x in X:
        dists = np.linalg.norm(prototypes - x, axis=1)
        nearest = np.argmin(dists)
        preds.append(proto_labels[nearest])
    return np.array(preds)
# --- Funkcja rysująca wykres ---

def plot_data():
    ax.clear()
    X = precomputed["X"]
    true_y = precomputed["y"]
    pred_y = precomputed["y_pred"]
    colors = precomputed["class_colors"]
    clf = precomputed["clf"]
    xx, yy, Z = precomputed["xx"], precomputed["yy"], precomputed["Z"]

    if X is None or clf is None:
        return

    # decision boundary
    if params["show_boundary"] and Z is not None:
        ax.contourf(xx, yy, Z, alpha=0.2)

    # dane
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=[colors[i] for i in true_y],
        s=20,
        linewidth=0
    )
    ax.scatter(
        X[:, 0],
        X[:, 1],
        facecolors='none',
        edgecolors=[colors[i] for i in pred_y],
        s=40,
        linewidths=2
    )

    # prototypy
    if params["show_proto"]:
        prototypes, proto_labels = clf.get_protos()
        colors = precomputed["class_colors"]

        for cls in np.unique(proto_labels):
            ax.scatter(
                prototypes[proto_labels == cls, 0],
                prototypes[proto_labels == cls, 1],
                marker='X',
                s=200,
                c=[colors[cls]],
                edgecolor='black',   # opcjonalnie kontur
                linewidth=1.2
            )

    # połączenia
    if params["show_edges"]:
        try:
            protos_id = np.array(sorted(set(sum(map(clf.proto_ensemble_.unpairCantor, clf.regions_), ()))))
            PX = clf.proto_ensemble_.proto[protos_id]
            protos_id_to_row = dict(zip(protos_id, range(len(protos_id))))
            for pair in clf.regions_:
                i, j = clf.proto_ensemble_.unpairCantor(pair)
                _x = [PX[protos_id_to_row[i]][0], PX[protos_id_to_row[j]][0]]
                _y = [PX[protos_id_to_row[i]][1], PX[protos_id_to_row[j]][1]]
                ax.plot(_x, _y, 'r')
        except:
            pass

    ax.set_title(f"Dataset: {params['dataset']} | APD")
    fig.canvas.draw_idle()

def plot_data2():
    ax_proto.clear()

    X = precomputed["X"]
    clf = precomputed["clf"]
    xx, yy = precomputed["xx"], precomputed["yy"]

    if X is None or clf is None:
        return

    prototypes, proto_labels = clf.get_protos()

    # =========================
    # tło decyzyjne prototypów
    # =========================
    if params["show_boundary"] and xx is not None:
        Z_proto = predict_by_prototypes(
            np.c_[xx.ravel(), yy.ravel()],
            prototypes,
            proto_labels
        ).reshape(xx.shape)

        ax_proto.contourf(xx, yy, Z_proto, alpha=0.25)

    # =========================
    # punkty (kolor = prototyp)
    # =========================
    preds = predict_by_prototypes(X, prototypes, proto_labels)
    colors = precomputed["class_colors"]
    c=[colors[i] for i in precomputed["y"]]
    ax_proto.scatter(
        X[:, 0],
        X[:, 1],
        c=[colors[i] for i in precomputed["y"]],
        alpha=0.8
    )

    # =========================
    # prototypy
    # =========================
    for cls in np.unique(proto_labels):
        ax_proto.scatter(
            prototypes[proto_labels == cls, 0],
            prototypes[proto_labels == cls, 1],
            marker='X',
            s=150,
            edgecolor='black'
        )
    ax_proto.set_title("Prototype assignment view")
    fig2.canvas.draw_idle()

def plot_data3():
    ax_regions.clear()

    X = precomputed["X"]
    clf = precomputed["clf"]
    xx, yy = precomputed["xx"], precomputed["yy"]

    if X is None or clf is None:
        return

    prototypes, proto_labels = clf.get_protos()

    # =========================
    # tło decyzyjne prototypów
    # =========================
    if params["show_boundary"] and xx is not None:
        from adversarial_prototype_decomposition.utils.apd_utils import get_region_assign_data
        qc = get_region_assign_data(clf.proto_ensemble_, np.c_[xx.ravel(), yy.ravel()])
        
        qcc = np.zeros(next(iter(qc.values())).shape)
        for i, k in enumerate(qc):
            qcc += (qc[k] * (i+1))
        qcc = qcc.reshape(xx.shape)
        ax_regions.contourf(xx, yy, qcc, alpha=0.25)

    # =========================
    # punkty (kolor = prototyp)
    # =========================
    preds = predict_by_prototypes(X, prototypes, proto_labels)
    colors = precomputed["class_colors"]
    c=[colors[i] for i in precomputed["y"]]
    ax_regions.scatter(
        X[:, 0],
        X[:, 1],
        c=[colors[i] for i in precomputed["y"]],
        cmap="viridis",
        alpha=0.8
    )

    # =========================
    # prototypy
    # =========================
    for cls in np.unique(proto_labels):
        ax_regions.scatter(
            prototypes[proto_labels == cls, 0],
            prototypes[proto_labels == cls, 1],
            marker='X',
            s=150,
            edgecolor='black'
        )
    if params["show_edges"]:
        try:
            protos_id = np.array(sorted(set(sum(map(clf.proto_ensemble_.unpairCantor, clf.regions_), ()))))
            PX = clf.proto_ensemble_.proto[protos_id]
            protos_id_to_row = dict(zip(protos_id, range(len(protos_id))))
            for pair in clf.regions_:
                i, j = clf.proto_ensemble_.unpairCantor(pair)
                _x = [PX[protos_id_to_row[i]][0], PX[protos_id_to_row[j]][0]]
                _y = [PX[protos_id_to_row[i]][1], PX[protos_id_to_row[j]][1]]
                ax_regions.plot(_x, _y, 'r')
        except Exception as e:
            print(e)

    ax_regions.set_title("Region assignment view")
    fig3.canvas.draw_idle()

# --- Tkinter GUI ---
root = tk.Tk()
root.title("Parametry danych + APD")
root.geometry("400x400")
frame = ttk.Frame(root, padding=10)
frame.pack(fill='both', expand=True)

# --- DATA ---
ttk.Label(frame, text="=== DATA ===").grid(column=0, row=0, columnspan=2, pady=5)
dataset_var = tk.StringVar(value="blobs")
ttk.Label(frame, text="Typ danych:").grid(column=0, row=1)
ttk.Combobox(frame, textvariable=dataset_var, values=("blobs", "moons", "circles", "gaussian", "multilabel"), state="readonly").grid(column=1, row=1)
samples_var = tk.StringVar(value="300")
ttk.Label(frame, text="Liczba próbek:").grid(column=0, row=2)
ttk.Entry(frame, textvariable=samples_var).grid(column=1, row=2)
classes_var = tk.StringVar(value="3")
ttk.Label(frame, text="Liczba klas:").grid(column=0, row=3)
ttk.Entry(frame, textvariable=classes_var).grid(column=1, row=3)
noise_var = tk.StringVar(value="0.1")
ttk.Label(frame, text="Szum:").grid(column=0, row=4)
ttk.Entry(frame, textvariable=noise_var).grid(column=1, row=4)

# --- APD ---
ttk.Label(frame, text="=== APD PARAMS ===").grid(column=0, row=5, columnspan=2, pady=5)
min_support_var = tk.StringVar(value="100")
ttk.Label(frame, text="min_support:").grid(column=0, row=6)
ttk.Entry(frame, textvariable=min_support_var).grid(column=1, row=6)
unbalanced_var = tk.StringVar(value="0.1")
ttk.Label(frame, text="unbalanced_rate:").grid(column=0, row=7)
ttk.Entry(frame, textvariable=unbalanced_var).grid(column=1, row=7)
prune_var = tk.BooleanVar(value=True)
ttk.Label(frame, text="prune_regions:").grid(column=0, row=8)
ttk.Checkbutton(frame, variable=prune_var).grid(column=1, row=8)
proto_var = tk.StringVar(value="3")
ttk.Label(frame, text="prototypy / klasa:").grid(column=0, row=9)
ttk.Entry(frame, textvariable=proto_var).grid(column=1, row=9)

# --- BUTTON ---
def update_params():
    params["dataset"] = dataset_var.get()
    params["n_samples"] = int(samples_var.get())
    params["n_classes"] = int(classes_var.get())
    params["noise"] = float(noise_var.get())
    params["min_support"] = int(min_support_var.get())
    params["unbalanced_rate"] = float(unbalanced_var.get())
    params["prune_regions"] = prune_var.get()
    params["n_proto"] = _update_n_proto(proto_var.get(), params["n_classes"])
    compute_apd()   # oblicz dane + APD
    print(f"DOKŁADNOŚĆ LVQ: {precomputed['proto_acc']}")
    plot_data()     # rysuj
    plot_data2()
    plot_data3()

def _update_n_proto(n_proto:str, n_class:int):
    res = []
    if "," in n_proto:
        for x in n_proto.split(","):
            res.append(int(x))
    else:
        for _ in range(n_class):
            res.append(int(n_proto))
    return res

ttk.Button(frame, text="Trenuj + wizualizuj", command=update_params).grid(column=0, row=10, columnspan=2, pady=10)

# --- Matplotlib CheckButtons ---
rax = plt.axes([0.05, 0.4, 0.2, 0.15])
check = CheckButtons(rax, ['Show Proto', 'Show Edges', 'Show Boundary'],
                     [params['show_proto'], params['show_edges'], params['show_boundary']])

def check_update(label):
    if label == 'Show Proto':
        params['show_proto'] = not params['show_proto']
    elif label == 'Show Edges':
        params['show_edges'] = not params['show_edges']
    elif label == 'Show Boundary':
        params['show_boundary'] = not params['show_boundary']
    plot_data()  # tylko rysowanie, bez przeliczania APD

check.on_clicked(check_update)

plt.show(block=False)
root.mainloop()
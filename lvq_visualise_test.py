# %% Imports
import pandas as pd
import numpy as np
from visualization import tsne_classification_pipeline, pca_classification_pipeline, umap_classification_pipeline
from sklvq import GLVQ, GMLVQ, LGMLVQ
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import numpy as np
from sklearn.manifold import TSNE

# %%Logger
class ProcessLogger:
    def __init__(self):
        self.states = np.array([])

    # A callback function has to accept two arguments, i.e., model and state, where model is the
    # current model, and state contains a number of the optimizers variables.
    def __call__(self, state):
        # print(state['variables'].shape)
        self.states = np.append(self.states, state)
        return False  # The callback function can also be used to stop training early,
        # if some condition is met by returning True.
# %% Func
def plot(X, y, Xp, yp):
    plt.figure(1, clear=True)
    x1 = X[:, 0]
    x2 = X[:, 1]
    plt.scatter(x1[y == 0],x2[y == 0],c="g", marker="*")
    plt.scatter(x1[y == 1],x2[y == 1],c="b", marker="o")

    xp1 = Xp[:, 0]
    xp2 = Xp[:, 1]
    plt.scatter(xp1[yp == 0],xp2[yp == 0],c="r",marker='*',s=100)
    plt.scatter(xp1[yp == 1],xp2[yp == 1],c="r",marker='o',s=100)
    plt.show()

def plot_steps_with_learning(X, y, Xps_lvq, yp_lvq, lc_lvq):
    from matplotlib.widgets import Slider

    n_steps = len(Xps_lvq)
    if len(lc_lvq) != n_steps:
        raise ValueError("Długość lc_lvq musi być taka sama jak liczba kroków Xps_lvq")

    fig, (ax_scatter, ax_curve) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)

    step = 0
    ax_scatter.set_title(f"Proto in dataset (step={step})")
    ax_scatter.set_xlabel("X1")
    ax_scatter.set_ylabel("X2")

    ax_scatter.scatter(X[y == 0, 0], X[y == 0, 1], c="g", marker="*", label="Class 0")
    ax_scatter.scatter(X[y == 1, 0], X[y == 1, 1], c="b", marker="o", label="Class 1")

    sc0 = ax_scatter.scatter([], [], c="r", marker="*", s=100, label="Proto class 0")
    sc1 = ax_scatter.scatter([], [], c="r", marker="o", s=100, label="Proto class 1")

    all_x = np.vstack([X] + list(Xps_lvq))
    ax_scatter.set_xlim(np.min(all_x[:, 0]) - 1, np.max(all_x[:, 0]) + 1)
    ax_scatter.set_ylim(np.min(all_x[:, 1]) - 1, np.max(all_x[:, 1]) + 1)

    ax_curve.set_title(f"Learning curve (value={lc_lvq[step]:.3f})")
    ax_curve.set_xlabel("Step")
    ax_curve.set_ylabel("Value")

    line_curve, = ax_curve.plot(range(n_steps), lc_lvq, '-o')
    point_curve, = ax_curve.plot([], [], 'ro')
    ax_curve.grid(True)
    ax_curve.set_xlim(-0.5, n_steps - 0.5)
    ax_curve.set_ylim(min(lc_lvq) * 0.95, max(lc_lvq) * 1.05)

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax=ax_slider, label="Step", valmin=0, valmax=n_steps - 1, valinit=0, valstep=1)

    def update(val):
        step = int(slider.val)
        Xp = Xps_lvq[step]
        sc0.set_offsets(np.c_[Xp[yp_lvq == 0, 0], Xp[yp_lvq == 0, 1]])
        sc1.set_offsets(np.c_[Xp[yp_lvq == 1, 0], Xp[yp_lvq == 1, 1]])
        point_curve.set_data([step], [lc_lvq[step]])
        ax_scatter.set_title(f"Proto in dataset (step={step})")
        ax_curve.set_title(f"Learning curve (value={lc_lvq[step]:.3f})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()

def plot_steps_with_learning_all(
    X, y,
    Xps_lvq, yp_lvq, lc_lvq,
    Xps_glvq, yp_glvq, lc_glvq,
    Xps_gmlvq, yp_gmlvq, lc_gmlvq,
    axis_x:tuple = None, axis_y:tuple = None
):
    from matplotlib.widgets import Slider

    n_steps = len(Xps_lvq)
    if axis_x is None or axis_y is None:
            axis_x = ("X1",0)
            axis_y = ("X2",1)
    if not (len(Xps_glvq) == len(Xps_gmlvq) == n_steps):
        raise ValueError("Wszystkie listy Xps_* muszą mieć tyle samo kroków")
    if not (len(lc_lvq) == len(lc_glvq) == len(lc_gmlvq) == n_steps):
        raise ValueError("Wszystkie listy lc_* muszą mieć tyle samo kroków")
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    plt.subplots_adjust(bottom=0.1, hspace=0.35, wspace=0.25)

    def setup_subplot(ax_s, ax_c, label, lc_values, axis_x=None, axis_y = None):
        
        ax_s.set_xlabel(axis_x[0])
        ax_s.set_ylabel(axis_y[0])
        ax_s.scatter(X[y == 0, axis_x[1]], X[y == 0, axis_y[1]], c="g", marker="*", label="Class 0")
        ax_s.scatter(X[y == 1, axis_x[1]], X[y == 1, axis_y[1]], c="b", marker="o", label="Class 1")
        sc0 = ax_s.scatter([], [], c="r", marker="*", s=100)
        sc1 = ax_s.scatter([], [], c="r", marker="o", s=100)

        all_x = np.vstack([X])
        ax_s.set_xlim(np.min(all_x[:, axis_x[1]]) - 1, np.max(all_x[:, axis_x[1]]) + 1)
        ax_s.set_ylim(np.min(all_x[:, axis_y[1]]) - 1, np.max(all_x[:, axis_y[1]]) + 1)

        ax_c.set_xlabel("Step")
        ax_c.set_ylabel("Value")
        line, = ax_c.plot(range(n_steps), lc_values, '-o')
        point, = ax_c.plot([], [], 'ro')
        ax_c.grid(True)
        ax_c.set_xlim(-0.5, n_steps - 0.5)
        ax_c.set_ylim(min(lc_values) * 0.95, max(lc_values) * 1.05)

        return sc0, sc1, line, point

    sc_lvq0, sc_lvq1, line_lvq, point_lvq = setup_subplot(
        axes[0, 0], axes[0, 1], "GLVQ", lc_lvq, axis_x, axis_y
    )
    sc_glvq0, sc_glvq1, line_glvq, point_glvq = setup_subplot(
        axes[1, 0], axes[1, 1], "GMLVQ", lc_glvq, axis_x, axis_y
    )
    sc_gmlvq0, sc_gmlvq1, line_gmlvq, point_gmlvq = setup_subplot(
        axes[2, 0], axes[2, 1], "LGMLVQ", lc_gmlvq, axis_x, axis_y
    )

    ax_slider = plt.axes([0.25, 0.03, 0.5, 0.03])
    slider = Slider(ax=ax_slider, label="Step", valmin=0, valmax=n_steps - 1, valinit=0, valstep=1)

    def update(val):
        step = int(slider.val)

        def update_subplot(ax_s, ax_c, sc0, sc1, point, Xps, yp, lc, label, ax, ay):
            Xp = Xps[step]
            sc0.set_offsets(np.c_[Xp[yp == 0, ax], Xp[yp == 0, ay]])
            sc1.set_offsets(np.c_[Xp[yp == 1, ax], Xp[yp == 1, ay]])
            point.set_data([step], [lc[step]])
            ax_s.set_title(f"{label} proto (step={step})")
            ax_c.set_title(f"{label} learning curve (value={lc[step]:.3f})")

        update_subplot(axes[0, 0], axes[0, 1], sc_lvq0, sc_lvq1, point_lvq, Xps_lvq, yp_lvq, lc_lvq, "GLVQ", axis_x[1],axis_y[1])
        update_subplot(axes[1, 0], axes[1, 1], sc_glvq0, sc_glvq1, point_glvq, Xps_glvq, yp_glvq, lc_glvq, "GMLVQ", axis_x[1],axis_y[1])
        update_subplot(axes[2, 0], axes[2, 1], sc_gmlvq0, sc_gmlvq1, point_gmlvq, Xps_gmlvq, yp_gmlvq, lc_gmlvq, "LGMLVQ", axis_x[1],axis_y[1])

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()

def tsne_transform(X, Xp):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Xp_scaled = scaler.transform(Xp)

    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X_scaled = pca.fit_transform(X_scaled)
        Xp_scaled = pca.transform(Xp_scaled)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        max_iter=1000,
        learning_rate=200,
        early_exaggeration=12,
        metric='euclidean'
    )

    X_all = np.vstack([X_scaled, Xp_scaled])
    X_all_tsne = tsne.fit_transform(X_all)
    
    X_tsne = X_all_tsne[:len(X)]
    Xp_tsne = X_all_tsne[len(X):]
    return X_tsne, Xp_tsne


def pca_transform(X, Xp=None, pca:PCA=None, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    if Xp is not None:
        Xp_scaled = scaler.transform(Xp)

    if pca is None:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X_scaled)
    if Xp is not None:
        Xp = pca.transform(Xp_scaled)

    return X, Xp, pca, scaler


def umap_transform(X, Xp=None, reducer=None, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    if Xp is not None:
        Xp_scaled = scaler.transform(Xp)

    if reducer is None:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=0.05,
            metric='euclidean'
        )

        X = reducer.fit_transform(X_scaled)
    if Xp is not None:
        Xp = reducer.transform(Xp_scaled)

    return X, Xp, reducer, scaler

# %%Load data
data = pd.read_csv("G:\Doktorat\Prototype-Pair-Ensemble\Data\data_UT1_v5_048.csv")
X_cols = ["Pressure - leak line","Temperature - leak line","Pressure - output","Temperature - suction line","Temperature - output","Flow - leak line","Flow - output","Sensor 1","Sensor 2","Sensor 3","Temp. diff"]
y_col = "stan"
X = data.loc[:, X_cols].values
y = data.loc[:,y_col].values
ohe = LabelEncoder()
y = ohe.fit_transform(y)
# %%Config params
N_PROTO = 5
STEP_SIZE = 0.1
MAX_RUNS = 100
BATCH_SIZE = 16
MODE = "NORMAL"
X_AXIS = (X_cols[2],2)
Y_AXIS = (X_cols[5],5)
NORMALIZE = True
# %%Create model
logger_glvq = ProcessLogger()
logger_gmlvq = ProcessLogger()
logger_lgmlvq = ProcessLogger()
scaler = StandardScaler()

model_glvq = GLVQ(prototype_n_per_class=np.array([N_PROTO, N_PROTO]),
             solver_params={"step_size": STEP_SIZE,
                            "max_runs": MAX_RUNS,
                            "batch_size": BATCH_SIZE,
                            "callback": logger_glvq,
                            })

model_gmlvq = GMLVQ(prototype_n_per_class=np.array([N_PROTO, N_PROTO]),
             solver_params={"step_size": STEP_SIZE,
                            "max_runs": MAX_RUNS,
                            "batch_size": BATCH_SIZE,
                            "callback": logger_gmlvq,
                            })

model_lgmlvq = LGMLVQ(prototype_n_per_class=np.array([N_PROTO, N_PROTO]),
             solver_params={"step_size": STEP_SIZE,
                            "max_runs": MAX_RUNS,
                            "batch_size": BATCH_SIZE,
                            "callback": logger_lgmlvq,
                            })
# %%Fit model
if NORMALIZE: X = scaler.fit_transform(X)
model_glvq.fit(X,y)
model_gmlvq.fit(X,y)
model_lgmlvq.fit(X,y)

yp_glvq = model_glvq.prototypes_labels_
yp_gmlvq = model_gmlvq.prototypes_labels_
yp_lgmlvq = model_lgmlvq.prototypes_labels_
# # %%Print Data
# # print(data)
# print(X[:,0].shape)
# print(y)
# print(model.prototypes_)
# print(model._prototypes_shape)
# # %% Tests
# iteration, fun = zip(*[(state["nit"], state["fun"]) for state in logger.states])
# plt.figure(2, clear=True)
# plt.title("Learning Curve (Less is better)")
# plt.plot(iteration, fun)
# plt.show()

# plt.figure(1, clear=True)
# x1 = X[:, 0]
# x2 = X[:, 1]
# plt.scatter(x1,x2,c=y)

# Xp = model.prototypes_
# xp1 = Xp[:, 0]
# xp2 = Xp[:, 1]
# plt.scatter(xp1,xp2,c="k",marker='x',s=100)

# plt.show()
# %%Plots
# plot(X,y, Xp, yp)

Xps_glvq = []
lc_glvq = []
Xps_gmlvq = []
lc_gmlvq = []
Xps_lgmlvq = []
lc_lgmlvq = []

if MODE == "PCA":
    X, _ , pca, scaler_pca = pca_transform(X)
if MODE == "UMAP":
    X, _ , umapt, scaler_pca = umap_transform(X)    

for var in logger_glvq.states:
    Xp = model_glvq.to_prototypes_view(var["variables"])
    if MODE == "PCA":
        _, Xp, _, _ = pca_transform(X, Xp, pca, scaler_pca)
    if MODE == "UMAP":
        _, Xp, _, _ = umap_transform(X, Xp, umapt, scaler_pca)
    Xps_glvq.append(Xp)
    lc_glvq.append(var["fun"])

for var in logger_gmlvq.states:
    Xp = model_gmlvq.to_prototypes_view(var["variables"])
    if MODE == "PCA":
        _, Xp, _, _ = pca_transform(X, Xp, pca, scaler_pca)
    if MODE == "UMAP":
        _, Xp, _, _ = umap_transform(X, Xp, umapt, scaler_pca)
    Xps_gmlvq.append(Xp)
    lc_gmlvq.append(var["fun"])

for var in logger_lgmlvq.states:
    Xp = model_lgmlvq.to_prototypes_view(var["variables"])
    if MODE == "PCA":
        _, Xp, _, _ = pca_transform(X, Xp, pca, scaler_pca)
    if MODE == "UMAP":
        _, Xp, _, _ = umap_transform(X, Xp, umapt, scaler_pca)
    Xps_lgmlvq.append(Xp)
    lc_lgmlvq.append(var["fun"])
    
# %%PLOTS2
plot_steps_with_learning_all(X,y, Xps_glvq, yp_glvq, lc_glvq,Xps_gmlvq, yp_gmlvq, lc_gmlvq,Xps_lgmlvq, yp_lgmlvq, lc_lgmlvq, X_AXIS,Y_AXIS)
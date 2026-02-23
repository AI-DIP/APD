import matplotlib.pyplot as plt
import pandas as pd
from ....apdlib import classifiers as apd
from sklearn.manifold import MDS

def get_plot_regions_centres(
    model: apd.APD_Classifier,
    name: str = "Proto",
    X_data=None,
    y_data=None,
    columns: list = None,
    x_axis: str = None,
    y_axis: str = None
):
    """
    Rysuje wykres punktów danych wraz z centrami regionów wyznaczonymi
    na podstawie prototypów klasyfikatora APD.

    Funkcja umożliwia wizualizację danych wejściowych (dla dwóch wskazanych
    osi cech), prototypów oraz wyznaczonych centrów regionów pomiędzy parami prototypów.

    Parametry
    ----------
    model : apd.APD_Classifier
        Wytrenowany klasyfikator APD, zawierający informacje o prototypach i regionach.
    name : str, domyślnie "Proto"
        Tytuł wykresu.
    X_data : ndarray, opcjonalnie
        Dane wejściowe (macierz cech). Jeśli podane, punkty zostaną narysowane.
    y_data : ndarray, opcjonalnie
        Etykiety klas odpowiadające danym wejściowym.
    columns : list
        Lista nazw cech odpowiadających kolumnom w `X_data`.
    x_axis : str
        Nazwa cechy przypisanej do osi X.
    y_axis : str
        Nazwa cechy przypisanej do osi Y.

    Zwraca
    -------
    matplotlib.figure.Figure
        Obiekt figury zawierający wykres.
    """
    # Tworzenie nowej figury i osi do rysowania wykresu
    fig, ax = plt.subplots(figsize=(6, 6))

    # Jeśli podano dane wejściowe i etykiety klas, narysuj punkty danych
    if X_data is not None and y_data is not None:
        # Wyciągnięcie wartości dla wskazanych osi
        X_dataX = X_data[:, columns.index(x_axis)]
        y_dataX = X_data[:, columns.index(y_axis)]

        # Podział danych na klasy 0 i 1
        X0_data = X_dataX[y_data == 0]
        y0_data = y_dataX[y_data == 0]    
        X1_data = X_dataX[y_data == 1]
        y1_data = y_dataX[y_data == 1]

        # Rysowanie punktów dla klasy 0
        ax.scatter(X0_data, y0_data, marker="o", color='blue', s=8, label="Class 0")
        # Rysowanie punktów dla klasy 1
        ax.scatter(X1_data, y1_data, marker="*", color='green', s=8, label="Class 1")

    # Rysowanie centrów regionów na podstawie prototypów
    if model is not None:
        # Odtworzenie współrzędnych prototypów ze skalowanych wartości
        proto_info = pd.DataFrame(model.scaler.inverse_transform(model.proto_ensemble_.proto), columns=columns)
        # Informacje o regionach (pary prototypów)
        region_info = pd.DataFrame(model.region_stats)

        # Rozpakowanie par prototypów (indeksów)
        pairsProto = list(zip(*region_info["Pair"].apply(model.proto_ensemble_.unpairCantor)))

        # Etykiety prototypów
        y = model.proto_ensemble_.proto_labels
        # Wartości współrzędnych prototypów na osiach
        XA = proto_info[x_axis].values
        yA = proto_info[y_axis].values

        # Obliczanie środków między parami prototypów
        XC = []
        yC = []
        for i in range(len(pairsProto[0])):
            XC.append((XA[pairsProto[0][i]] + XA[pairsProto[1][i]]) / 2.0)
            yC.append((yA[pairsProto[0][i]] + yA[pairsProto[1][i]]) / 2.0)

        # Rysowanie centrów regionów
        ax.scatter(XC, yC, marker="+", color='red', s=80, label="Region Center")

    # Opisy osi i tytuł wykresu
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{name}")

    # Dodanie legendy i siatki
    ax.legend()
    ax.grid(True)

    # Zwrócenie obiektu figury (można go dalej wykorzystać lub zapisać do pliku)
    return fig

def get_plot(
    model: apd.APD_Classifier,
    name: str = "Proto",
    X_data=None,
    y_data=None,
    columns: list = None,
    x_axis: str = None,
    y_axis: str = None
):
    """
    Rysuje wykres punktów danych wraz z prototypami klasyfikatora APD
    oraz powiązaniami pomiędzy parami prototypów.

    Funkcja umożliwia wizualizację danych wejściowych (dla dwóch wskazanych
    cech), prototypów przypisanych do klas oraz linii łączących prototypy
    wchodzące w skład tych samych par regionów.
    """
    # Tworzenie figury i osi wykresu
    fig, ax = plt.subplots(figsize=(6, 6))

    # Rysowanie punktów danych wejściowych, jeśli są dostępne
    if X_data is not None and y_data is not None:
        # Wyciągnięcie wartości dla wybranych osi
        X_dataX = X_data[:, columns.index(x_axis)]
        y_dataX = X_data[:, columns.index(y_axis)]

        # Podział danych na klasy 0 i 1
        X0_data = X_dataX[y_data == 0]
        y0_data = y_dataX[y_data == 0]    
        X1_data = X_dataX[y_data == 1]
        y1_data = y_dataX[y_data == 1]

        # Rysowanie punktów klasy 0
        ax.scatter(X0_data, y0_data, marker="o", color='blue', s=8, label="Class 0")
        # Rysowanie punktów klasy 1
        ax.scatter(X1_data, y1_data, marker="*", color='green', s=8, label="Class 1")

    # Rysowanie prototypów i par prototypów
    if model is not None:
        # Przywrócenie oryginalnej skali prototypów
        proto_info = pd.DataFrame(
            model.scaler.inverse_transform(model.proto_ensemble_.proto),
            columns=columns
        )
        # Informacje o regionach (pary prototypów)
        region_info = pd.DataFrame(model.region_stats)

        # Rozpakowanie par prototypów (indeksy prototypów połączonych w pary)
        pairsProto = list(zip(*region_info["Pair"].apply(model.proto_ensemble_.unpairCantor)))

        # Etykiety prototypów
        y = model.proto_ensemble_.proto_labels
        # Wartości prototypów dla wskazanych osi
        XA = proto_info[x_axis].values
        yA = proto_info[y_axis].values

        # Podział prototypów na klasy
        X0 = XA[y == 0]
        y0 = yA[y == 0]
        X1 = XA[y == 1]
        y1 = yA[y == 1]

        # Rysowanie prototypów dla obu klas
        ax.scatter(X0, y0, marker="o", s=80, color='red', label="Proto Class 0")
        ax.scatter(X1, y1, marker="*", s=80, color='red', label="Proto Class 1")

        # Tworzenie listy par prototypów (ich współrzędnych)
        pairs = []
        for i in range(len(pairsProto[0])):
            pairs.append(
                (XA[pairsProto[0][i]], yA[pairsProto[0][i]], XA[pairsProto[1][i]], yA[pairsProto[1][i]])
            )

        # Rysowanie linii łączących prototypy w pary
        for i, (x0, y0, x1, y1) in enumerate(pairs):
            if i == 0:
                # Dodanie etykiety tylko do pierwszej linii
                ax.plot([x0, x1], [y0, y1], color='red', label='Pair')
            ax.plot([x0, x1], [y0, y1], color='red')

    # Ustawienia osi, tytułu i legendy
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{name}")
    ax.legend()
    ax.grid(True)

    # Zwrócenie figury z wykresem
    return fig

def get_prototypes_plot_MDS(
    model: apd.APD_Classifier,
    name: str = "Proto",
    X_data=None,
    y_data=None
):
    """
    Rysuje wykres prototypów klasyfikatora APD w przestrzeni 2D
    uzyskanej metodą MDS (Multidimensional Scaling).

    Funkcja umożliwia wizualizację zarówno danych wejściowych (opcjonalnie),
    jak i prototypów oraz powiązań pomiędzy prototypami w przestrzeni
    o zmniejszonej wymiarowości.

    Parametry
    ----------
    model : apd.APD_Classifier
        Wytrenowany klasyfikator APD, zawierający prototypy i informacje o regionach.
    name : str, domyślnie "Proto"
        Tytuł wykresu.
    X_data : ndarray, opcjonalnie
        Dane wejściowe (macierz cech). Jeśli podane, zostaną odwzorowane w 2D i narysowane.
    y_data : ndarray, opcjonalnie
        Etykiety klas odpowiadające danym wejściowym.

    Zwraca
    -------
    matplotlib.figure.Figure
        Obiekt figury zawierający wykres MDS.
    """
    # Tworzenie figury i osi wykresu
    fig, ax = plt.subplots(figsize=(6, 6))

    # Inicjalizacja MDS w wersji niemetrcznej
    mds = MDS(
        n_components=2,
        dissimilarity='euclidean',
        random_state=42,
        normalized_stress=True,
        metric=False
    )

    # Próba odwzorowania danych wejściowych, jeśli zostały podane
    try:
        if X_data is not None and y_data is not None:
            X_data = mds.fit_transform(X_data)
            # Podział punktów według klas
            X0_data = X_data[y_data == 0]
            X1_data = X_data[y_data == 1]
            # Rysowanie punktów danych wejściowych
            ax.scatter(X0_data[:, 0], X0_data[:, 1], marker="o", color='blue', s=8, label="Class 0")
            ax.scatter(X1_data[:, 0], X1_data[:, 1], marker="*", color='green', s=8, label="Class 1")
    except Exception:
        # Obsługa sytuacji, gdy MDS nie może zostać dopasowany do wszystkich danych
        print("Cannot create MDS for all X data")
        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)

    # Pobranie prototypów i ich etykiet
    X = model.proto_ensemble_.proto
    y = model.proto_ensemble_.proto_labels

    # Odwzorowanie prototypów do przestrzeni 2D
    X = mds.fit_transform(X)
    X0 = X[y == 0]
    X1 = X[y == 1]

    # Pobranie informacji o regionach (parach prototypów)
    region_info = pd.DataFrame(model.region_stats)
    pairsProto = list(zip(*region_info["Pair"].apply(model.proto_ensemble_.unpairCantor)))

    # Tworzenie listy współrzędnych dla par prototypów
    pairs = []
    for i in range(len(pairsProto[0])):
        pairs.append((
            X[pairsProto[0][i]][0],
            X[pairsProto[0][i]][1],
            X[pairsProto[1][i]][0],
            X[pairsProto[1][i]][1]
        ))

    # Rysowanie prototypów obu klas
    ax.scatter(X0[:, 0], X0[:, 1], marker="o", s=80, color='blue', label="Proto Class 0")
    ax.scatter(X1[:, 0], X1[:, 1], marker="*", s=80, color='green', label="Proto Class 1")

    # Rysowanie połączeń między parami prototypów
    for i, (x0, y0, x1, y1) in enumerate(pairs):
        if i == 0:
            # Dodanie etykiety tylko do pierwszej linii
            ax.plot([x0, x1], [y0, y1], color='red', label='Pair')
        ax.plot([x0, x1], [y0, y1], color='red')

    # Podpisy osi i tytuł zawierający wartość stresu MDS
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.set_title(f"{name} MDS (stress={mds.stress_:.2f})")

    # Dodanie legendy i siatki
    ax.legend()
    ax.grid(True)

    # Zwrócenie figury
    return fig

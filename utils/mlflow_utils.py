import mlflow
import pandas as pd
import os

def save_pandas_as_artefact(
    data: pd.DataFrame,
    name: str,
    path: str = None,
    index_name: str = "Id"
):
    """
    Zapisuje obiekt DataFrame jako plik CSV i rejestruje go jako artefakt w MLflow.

    Parametry
    ----------
    data : pd.DataFrame
        Ramka danych do zapisania.
    name : str
        Nazwa pliku (np. 'dane.csv').
    path : str, opcjonalnie
        Ścieżka do katalogu, w którym zostanie zapisany plik.
    index_name : str, domyślnie "Id"
        Nazwa kolumny indeksu w pliku CSV.

    Zwraca
    -------
    None
    """
    # Budowanie pełnej ścieżki do pliku
    full_path = os.path.join(path, name)
    # Zapisanie DataFrame do pliku CSV z indeksem
    data.to_csv(full_path, index=True, index_label=index_name)
    # Rejestracja pliku jako artefaktu w MLflow
    mlflow.log_artifact(full_path)


def save_fig_as_artefact(fig, name: str, path: str):
    """
    Zapisuje wykres matplotlib jako plik PNG i rejestruje go jako artefakt w MLflow.

    Parametry
    ----------
    fig : matplotlib.figure.Figure
        Obiekt wykresu do zapisania.
    name : str
        Nazwa pliku (np. 'wykres.png').
    path : str
        Ścieżka do katalogu, w którym zostanie zapisany plik.

    Zwraca
    -------
    None
    """
    # Budowanie pełnej ścieżki do pliku
    full_path = os.path.join(path, name)
    # Zapisanie wykresu do pliku PNG o wysokiej rozdzielczości
    fig.savefig(full_path, dpi=600, bbox_inches='tight', pad_inches=1.0)
    # Rejestracja wykresu jako artefaktu w MLflow (plik PNG)
    mlflow.log_figure(fig, f"{name}.png")

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path

@pytest.fixture
def processed_data():
    # Ruta absoluta al archivo parquet
    file_path = Path(__file__).resolve().parent / "../data/processed/data_processed.parquet"

    # Verificar si el archivo existe
    if not file_path.exists():
        pytest.fail(f"Archivo no encontrado: {file_path}")

    return pd.read_parquet(file_path)

# Prueba: Verificar duplicados en columnas clave
def test_no_duplicates(processed_data: pd.DataFrame):
    assert processed_data['account_number'].is_unique, "Hay valores duplicados en 'account_number'"
    assert not processed_data.duplicated(subset=['account_number', 'user_id']).any(), (
        "Hay filas duplicadas en la combinación de 'account_number' y 'user_id'"
    )

# Prueba: Verificar valores nulos en columnas no nulas
def test_no_null_values(processed_data: pd.DataFrame):
    non_nullable_columns = ['account_number', 'user_id', 'transaction_date', 'transaction_amount']
    for column in non_nullable_columns:
        assert processed_data[column].notnull().all(), f"La columna '{column}' contiene valores nulos"

# Prueba: Verificar rango de fechas
def test_date_range(processed_data: pd.DataFrame):
    min_date = datetime(2020, 1, 1)
    max_date = datetime(2025, 12, 31)
    assert processed_data['transaction_date'].min() >= min_date, (
        f"Hay transacciones con fechas menores a {min_date}"
    )
    assert processed_data['transaction_date'].max() <= max_date, (
        f"Hay transacciones con fechas mayores a {max_date}"
    )

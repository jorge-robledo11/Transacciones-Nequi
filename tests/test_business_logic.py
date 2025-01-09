import pytest
import pandas as pd
from pathlib import Path

# Fixture para cargar los datos procesados
@pytest.fixture
def processed_data():
    # Ruta absoluta al archivo parquet
    file_path = Path(__file__).resolve().parent / "../data/processed/data_processed.parquet"

    # Verificar si el archivo existe
    if not file_path.exists():
        pytest.fail(f"Archivo no encontrado: {file_path}")

    return pd.read_parquet(file_path)

# Prueba: Verificar relación entre usuarios y cuentas
# def test_user_account_relationship(processed_data: pd.DataFrame):
#     user_to_account_counts = processed_data.groupby('user_id')['account_number'].nunique()
#     assert (user_to_account_counts == 1).all(), (
#         "Hay usuarios asociados a múltiples cuentas"
#     )

# Prueba: Verificar que todos los montos sean positivos
def test_transaction_amount_positive(processed_data: pd.DataFrame):
    assert (processed_data['transaction_amount'] > 0).all(), (
        "Hay transacciones con montos negativos en 'transaction_amount'"
    )


import pytest
from pandera.errors import SchemaError
from datetime import datetime
import pandas as pd
from pandera import DataFrameSchema, Column, Check
from pathlib import Path

# Esquema de validación
schema = DataFrameSchema({
    'account_number': Column(str, Check.str_length(10, 30), nullable=False),
    'user_id': Column(str, Check.str_length(36), nullable=False),
    'transaction_date': Column(datetime, nullable=False),
    'transaction_amount': Column(float, Check.greater_than(0), nullable=False),
    'transaction_type': Column(
        str, 
        Check.isin(['debit', 'credit']), 
        nullable=False
    ),
    'fraction_flag': Column(
        str, 
        Check.isin(['fraccionada', 'no fraccionada']), 
        nullable=False
    ),
})

@pytest.fixture
def processed_data():
    # Ruta absoluta al archivo parquet
    file_path = Path(__file__).resolve().parent / "../data/processed/data_processed.parquet"

    # Verificar si el archivo existe
    if not file_path.exists():
        pytest.fail(f"Archivo no encontrado: {file_path}")

    return pd.read_parquet(file_path)

# Prueba: Validar datos procesados con Pandera
def test_validate_schema(processed_data: pd.DataFrame):
    try:
        validated_data = schema.validate(processed_data)
        assert not validated_data.empty, "El DataFrame validado no debe estar vacío"
    except SchemaError as e:
        pytest.fail(f"Falló la validación del esquema: {e}")

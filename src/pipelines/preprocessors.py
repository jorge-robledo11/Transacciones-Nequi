import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NumericColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Un transformador que convierte exclusivamente columnas con valores numéricos almacenados como cadenas (str) 
    a tipos numéricos adecuados (float), sin afectar columnas no numéricas.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X_copy = X.copy()

        for col in X_copy.columns:
            if X_copy[col].dtype == object:  # Verificar si la columna es de tipo str
                # Verificar si los valores en la columna son puramente numéricos
                if X_copy[col].apply(lambda x: str(x).replace('.', '', 1).isdigit()).all():
                    # Convertir a float si todos los valores son numéricos
                    X_copy[col] = X_copy[col].astype(float)
        return X_copy


class DropDuplicateColumnsTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer to drop columns in a DataFrame if their values are identical to another column.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    columns_to_drop_: list
        A list to store the names of columns that were dropped during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training, it returns itself unchanged.

    transform(X):
        Remove columns from the input DataFrame X if their values are identical to another column.

    Examples:
    ---------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3], 'C': [4, 5, 6]})
    >>> transformer = DropDuplicateColumnsTransformer()
    >>> transformed_data = transformer.transform(data)
    >>> transformed_data
        A  C
    0  1  4
    1  2  5
    2  3  6
    """

    def __init__(self):
        self.columns_to_drop_ = list()

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X_copy = X.copy()
        columns_to_drop = []
        for i, col1 in enumerate(X_copy.columns):
            for col2 in X_copy.columns[i + 1:]:
                if X_copy[col1].equals(X_copy[col2]):
                    columns_to_drop.append(col2)
        self.columns_to_drop_ = columns_to_drop
        X_transformed = X_copy.drop(columns=columns_to_drop)
        return X_transformed


class ColumnsRenameTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer for renaming columns of a DataFrame using a custom transformation function.

    Parameters:
    -----------
    transformation: function
        A function that takes a column name (string) as input and returns the new column name.

    Attributes:
    -----------
    transformation: function
        The transformation function used for renaming column names.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

    transform(X):
        Rename columns of the input DataFrame X using the provided transformation function.

    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> def custom_transform(col_name):
    ...     return col_name.lower()
    >>> transformer = ColumnsRenameTransformer(transformation=custom_transform)
    >>> df_transformed = transformer.transform(df)
    >>> df_transformed
       a  b
    0  1  3
    1  2  4
    """

    def __init__(self, transformation):
        
        """
        Initialize the transformer with a custom column name transformation function.

        Parameters:
        ----------
        transformation : function
            A function that takes a column name (string) as input and returns the new column name.
        """
        
        self.transformation = transformation

    def fit(self, X:pd.DataFrame, y=None):
        
        """
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame.

        y: None
            Ignored. This parameter is included for compatibility with scikit-learn's transformers.

        Returns:
        --------
        self : ColumnNameTransformer
            The fitted transformer instance.
        """
        
        return self

    def transform(self, X:pd.DataFrame):
        
        """
        Rename columns of the input DataFrame X using the provided transformation function.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame with columns to be renamed.

        Returns:
        --------
        X_transformed: pandas.DataFrame
            The DataFrame with column names transformed according to the provided function.
        """
        
        X_transformed = X.rename(columns=self.transformation)
        return X_transformed


class DropDuplicatedRowsTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer to remove duplicate rows from a DataFrame.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    None

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

    transform(X):
        Remove duplicate rows from the input DataFrame X.

    Examples:
    --------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris(as_frame=True)
    >>> df = iris.data
    >>> transformer = DropDuplicatedRowsTransformer()
    >>> df_no_duplicates = transformer.transform(df)
    """

    def __init__(self):
        
        """
        Initialize the transformer.

        Parameters:
        -----------
        None
        """
        
        pass

    def fit(self, X:pd.DataFrame, y=None):
        
        """
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame.

        y: None
            Ignored. This parameter is included for compatibility with scikit-learn's transformers.

        Returns:
        --------
        self : DropDuplicatedTransformer
            The fitted transformer instance.
        """
        
        return self

    def transform(self, X:pd.DataFrame):
        
        """
        Remove duplicate rows from the input DataFrame X.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame from which duplicate rows will be removed.

        Returns:
        --------
        X_no_duplicates: pandas.DataFrame
            The DataFrame with duplicate rows removed.
        """
        
        X = X.copy()
        X_no_duplicates = X.drop_duplicates(ignore_index=True)
        return X_no_duplicates


class FillMissingValuesTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer to fill missing values in a DataFrame with np.nan.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    None

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

    transform(X):
        Fill missing values in the input DataFrame X with NaN.

    Examples:
    --------
    >>> data = pd.DataFrame({'col1': ['A', 'B', '', 'C'], 'col2': [1, np.nan, 'None', 'N/A']})
    >>> transformer = FillMissingValuesTransformer()
    >>> data_no_missing = transformer.transform(data)
    """

    def __init__(self):
        
        """
        Initialize the transformer.

        Parameters:
        -----------
        None
        """
        
        pass

    def fit(self, X:pd.DataFrame, y=None):
        
        """
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame.

        y: None
            Ignored. This parameter is included for compatibility with scikit-learn's transformers.

        Returns:
        --------
        self: FillMissingValuesTransformer
            The fitted transformer instance.
        """
        
        return self

    def transform(self, X:pd.DataFrame):
        
        """
        Fill missing values in the input DataFrame X with np.nan.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame in which missing values will be replaced with np.nan.

        Returns:
        --------
        X_no_missing: pandas.DataFrame
            The DataFrame with missing values replaced by np.nan.
        """
        
        X = X.copy()
        X_no_missing = X.fillna(np.nan)
        X_no_missing = X_no_missing.replace({'ERROR': np.nan,
                                             '': np.nan,
                                             'None': np.nan,
                                             'n/a': np.nan,
                                             'N/A': np.nan,
                                             'NULL': np.nan, 
                                             'NA': np.nan,
                                             'NAN': np.nan})
        return X_no_missing


class DateColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to detect and transform columns containing date-like values to datetime type.
    """

    def __init__(self, sample_size=10, date_formats=None):
        """
        Parameters:
        -----------
        sample_size: int, optional (default=10)
            Number of random samples to test for datetime-like patterns.
        
        date_formats: list of str, optional (default=None)
            A list of formats to check for datetime conversion.
        """
        self.sample_size = sample_size
        self.date_formats = date_formats or [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f', '%d-%b-%Y', '%d-%B-%Y'
        ]
        self.date_columns = list()

    def fit(self, X: pd.DataFrame, y=None):
        """
        Identify columns that should be transformed to datetime.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame.

        y: None
            Ignored.

        Returns:
        --------
        self: DateColumnsTransformer
            The fitted transformer.
        """
        self.date_columns = list()

        for col in X.columns:
            if X[col].isnull().all():
                # Skip columns that are completely null
                continue

            sample = X[col].dropna().sample(
                n=min(self.sample_size, len(X[col].dropna())),
                random_state=42
            )

            for fmt in self.date_formats:
                try:
                    # Attempt to parse the sample using the format
                    pd.to_datetime(sample, format=fmt, errors='raise')
                    self.date_columns.append((col, fmt))
                    break
                except (ValueError, TypeError):
                    continue

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform identified columns to datetime.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame.

        Returns:
        --------
        X_transformed: pandas.DataFrame
            The transformed DataFrame with datetime columns.
        """
        X_transformed = X.copy()

        for col, fmt in self.date_columns:
            X_transformed[col] = pd.to_datetime(
                X_transformed[col],
                format=fmt,
                errors='coerce'
            )

        return X_transformed

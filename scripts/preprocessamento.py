import pandas as pd
import re

def preprocess_data(df, cols_to_drop=None):
    """
    Pipeline para pré-processar texto em um DataFrame.
    
    Args:
    - df (pd.DataFrame): DataFrame com os dados.
    - cols_to_drop (list): Lista de colunas a serem removidas.
    
    Returns:
    - pd.DataFrame: DataFrame pré-processado.
    """
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(lower)
        df[col] = df[col].apply(remove_urls)
        df[col] = df[col].apply(remove_chars_estranhos)
        df[col] = df[col].apply(remove_digitos)
    
    return df

def lower(text):
    """Converte texto para letras minúsculas."""
    return text.lower()

def remove_chars_estranhos(text):
    """Remove caracteres não alfanuméricos, mantendo espaços."""
    return re.sub(r'[^\w\s]', '', text)

def remove_digitos(text):
    """Remove dígitos numéricos do texto."""
    return re.sub(r'\d', '', text)

def remove_urls(text):
    """Remove URLs do texto."""
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.sub('', text)
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download()

def pipeline_processar_texto(text):
    """
    Função completa de pré-processamento para aplicar a cada linha de texto.
    """
    text = lower(text)
    text = remove_urls(text)
    text = remove_chars_estranhos(text)
    text = remove_digitos(text)
    tokens = tokenizar_texto(text)
    tokens = remover_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens

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

def tokenizar_texto(text):
    """Tokeniza o texto em palavras individuais."""
    return word_tokenize(text)

def remover_stopwords(tokens):
    """Remove stopwords de uma lista de tokens."""
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens):
    """
    Aplica lematização em uma lista de tokens.
    
    Args:
    - tokens (list): Lista de palavras/token.
    
    Returns:
    - list: Lista de palavras lematizadas.
    """
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(word):
        """Mapeia a tag do POS do nltk para o formato esperado pelo WordNetLemmatizer."""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import string
import warnings

# Ignorar warnings para manter a saída limpa
warnings.filterwarnings('ignore')

# Certifique-se de que as stopwords do português estão baixadas (rode apenas uma vez)
try:
    stopwords_pt = stopwords.words('portuguese')
except LookupError:
    # Se der erro, baixa
    nltk.download('stopwords')
    stopwords_pt = stopwords.words('portuguese')

# Nome do arquivo CSV BRUTO que acabamos de criar
NOME_ARQUIVO_BRUTO = 'tweets_hb20_onix_2000_brutos_V4_Unico.csv'

# Carregar o DataFrame
try:
    df = pd.read_csv(NOME_ARQUIVO_BRUTO, encoding='utf-8')
    print(f"✅ Arquivo bruto '{NOME_ARQUIVO_BRUTO}' carregado com sucesso.")
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo '{NOME_ARQUIVO_BRUTO}' não encontrado. Certifique-se de que a Versão 4.0 do script foi executada e salvou o arquivo.")
    exit()

# ----------------------------------------------------
# 1. FUNÇÕES DE PRÉ-PROCESSAMENTO (CURADORIA)
# ----------------------------------------------------

def clean_text(text):
    """Executa a sequência de limpeza em um tweet."""
    
    # 1. Transformar para minúsculas e remover espaços extras
    text = str(text).lower().strip()
    
    # 2. Remoção de Ruído Estrutural (URLs, Mentions, Hashtags)
    # Remove URLs (http/https)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove Mentions (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove Hashtags (#topic - mantendo a palavra após o # se desejar, mas aqui removemos)
    text = re.sub(r'#\w+', '', text)
    # Remove caracteres especiais HTML/outros
    text = re.sub(r'&amp;', '', text)
    
    # 3. Remoção de Pontuação, Números e Emojis
    # Remove pontuação (usa a lista de pontuações padrão e substitui por espaço)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove números
    text = re.sub(r'\d+', '', text)
    # Remove caracteres não-alfabéticos (útil para emojis e símbolos)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Tokenização e Stopwords
    # Remove espaços duplos criados pela limpeza
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords_func(text):
    """Remove stopwords do texto (opcional na Curadoria Bruta, mas crucial para o modelo)."""
    tokens = text.split()
    tokens_filtered = [word for word in tokens if word not in stopwords_pt]
    return ' '.join(tokens_filtered)

# ----------------------------------------------------
# 2. APLICAÇÃO DA LIMPEZA
# ----------------------------------------------------

print("\n--- Iniciando o Pré-processamento (Curadoria) ---")

# Criar uma nova coluna para o texto limpo
df['content_limpo'] = df['content'].apply(clean_text)

# (Opcional, mas recomendado para análise) Aplicar remoção de Stopwords
df['content_processado'] = df['content_limpo'].apply(remove_stopwords_func)

print("--- Pré-processamento Concluído ---")

# ----------------------------------------------------
# 3. VERIFICAÇÃO E SALVAMENTO
# ----------------------------------------------------

# Exibir comparação antes e depois
print("\n--- Amostra de Comparação (Bruto vs. Processado) ---")
df_comparacao = df[['content', 'content_limpo', 'content_processado']].head(10)
pd.set_option('display.max_colwidth', None)
print(df_comparacao)
pd.set_option('display.max_colwidth', 50) # Reset

# Salvar o arquivo processado
NOME_ARQUIVO_FINAL = 'tweets_hb20_onix_2000_processado.csv'
df.to_csv(NOME_ARQUIVO_FINAL, index=False, encoding='utf-8')

print(f"\n✅ Curadoria Finalizada com sucesso!")
print(f"Arquivo processado salvo como: '{NOME_ARQUIVO_FINAL}'")

# ----------------------------------------------------
# 4. PRÓXIMO PASSO
# ----------------------------------------------------
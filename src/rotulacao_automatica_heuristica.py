import pandas as pd
import random

# Nome do arquivo CSV PROCESSADO da etapa anterior
NOME_ARQUIVO_PROCESSADO = 'tweets_hb20_onix_2000_processado.csv'

# Carregar o DataFrame processado
try:
    df = pd.read_csv(NOME_ARQUIVO_PROCESSADO, encoding='utf-8')
    # Usaremos a coluna limpa para a rotulação
    TEXT_COLUMN = 'content_limpo'
    print(f"✅ Arquivo processado '{NOME_ARQUIVO_PROCESSADO}' carregado com sucesso.")
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo '{NOME_ARQUIVO_PROCESSADO}' não encontrado. Certifique-se de que a etapa anterior foi executada.")
    exit()

# ----------------------------------------------------
# 1. LÉXICO HEURÍSTICO (SIMULAÇÃO)
# ----------------------------------------------------
# Palavras-chave retiradas das listas de vocabulário do script V4.0
PALAVRAS_POSITIVAS = [
    'irado', 'sensacional', 'massa', 'topzera', 'chave', 'daora', 
    'lindo', 'voando', 'zero', 'valeu', 'melhor', 'frente', 'garantia', 
    'design', 'potencia', 'economia', 'conforto', 'liquidez', 'mylink', 
    'seguranca', 'dirigibilidade', 'eficiencia', 'espaco', 'airbags'
]

PALAVRAS_NEGATIVAS = [
    'ranço', 'horrivel', 'zuado', 'mico', 'dor', 'cabeca', 'quebradeira', 
    'lixo', 'vergonha', 'medo', 'ridiculo', 'caro', 'travando', 'ruim',
    'barulho', 'falha', 'consumo', 'problema', 'lento', 'descascando', 
    'suspensao'
]

# Palavras-chave do Ruído (Irrelevantes para o sentimento)
PALAVRAS_RUIDO = [
    'padaria', 'pao', 'futebol', 'vizinho', 'cor', 'uniforme', 'pizza',
    'estudar', 'tcc', 'ajudar', 'uber', 'parou', 'cidade', 'promocao'
]

# ----------------------------------------------------
# 2. FUNÇÃO DE ROTULAÇÃO AUTOMÁTICA
# ----------------------------------------------------

def automatic_labeling(text):
    """Atribui um rótulo de sentimento baseado em palavras-chave."""
    
    # Converte o texto para um conjunto de palavras para busca rápida
    words = set(text.split())
    
    # Contagem de ocorrências
    positive_count = len(words.intersection(PALAVRAS_POSITIVAS))
    negative_count = len(words.intersection(PALAVRAS_NEGATIVAS))
    ruido_count = len(words.intersection(PALAVRAS_RUIDO))
    
    # --- HEURÍSTICA DE CLASSIFICAÇÃO ---
    
    # 1. Ruído/Irrelevante (Prioridade baixa, pois se tiver um termo forte, pode não ser ruído)
    if ruido_count >= 1 and positive_count == 0 and negative_count == 0:
        return 'Neutro/Ruído' # Tweets que falam de pizza, futebol, etc.

    # 2. Positivo Forte
    if positive_count > negative_count and positive_count >= 2:
        return 'Positivo'
    
    # 3. Negativo Forte
    if negative_count > positive_count and negative_count >= 2:
        return 'Negativo'
    
    # 4. Neutro/Dúvida (Onde a contagem é igual, baixa, ou quando há ironia/conflito)
    if positive_count == negative_count or (positive_count == 1 and negative_count == 0) or (positive_count == 0 and negative_count == 1):
        # Estes são os casos mais ambíguos: pode ser ironia, dúvida ou apenas menção fraca.
        # Na ausência de rotulação manual, classificamos como Neutro.
        return 'Neutro/Conflito'
    
    # Caso padrão (texto sem nenhuma palavra-chave forte)
    return 'Neutro/Ruído'


# ----------------------------------------------------
# 3. APLICAÇÃO DA ROTULAGEM
# ----------------------------------------------------

print("\n--- Iniciando a Rotulação Automática ---")

# Aplica a função de labeling à coluna limpa
df['sentiment_label'] = df[TEXT_COLUMN].apply(automatic_labeling)

print("--- Rotulação Concluída ---")

# ----------------------------------------------------
# 4. VERIFICAÇÃO E SALVAMENTO
# ----------------------------------------------------

# Exibir a distribuição dos rótulos
print("\n--- Distribuição dos Rótulos Gerados ---")
print(df['sentiment_label'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Exibir amostras para verificação
print("\n--- Amostra de Comparação (Limpo vs. Rótulo) ---")
df_comparacao = df[['content_limpo', 'sentiment_label']].sample(5)
pd.set_option('display.max_colwidth', None)
print(df_comparacao)
pd.set_option('display.max_colwidth', 50) # Reset

# Salvar o arquivo rotulado
NOME_ARQUIVO_ROTULADO = 'tweets_hb20_onix_2000_rotulado.csv'
df.to_csv(NOME_ARQUIVO_ROTULADO, index=False, encoding='utf-8')

print(f"\n✅ Rotulação Heurística Finalizada com sucesso!")
print(f"Arquivo rotulado salvo como: '{NOME_ARQUIVO_ROTULADO}'")

# ----------------------------------------------------
# 5. PRÓXIMO PASSO: MODELAGEM
# ----------------------------------------------------
# Arquivo: rotulacao_automatica_heuristica.py

def rotular_texto_heuristica(texto_limpo):
    """
    Rotula o sentimento de um texto limpo (sem stopwords) usando uma abordagem heurística
    baseada em léxicos simples.

    Args:
        texto_limpo (str): O texto pré-processado (minusculo, sem pontuacao, sem stopwords).

    Returns:
        str: 'POSITIVO', 'NEGATIVO', ou 'NEUTRO'.
    """
    
    # 💡 LÉXICO SIMPLES EM PORTUGUÊS
    
    # Palavras Positivas Comuns
    lexico_pos = set([
        'bom', 'boa', 'ótimo', 'excelente', 'fantástico', 'perfeito', 
        'lindo', 'confortável', 'econômico', 'eficiente', 'agradável', 
        'top', 'sensacional', 'incrível', 'gostei', 'recomendo', 'melhor'
    ])

    # Palavras Negativas Comuns
    lexico_neg = set([
        'ruim', 'péssimo', 'lento', 'quebra', 'defeito', 'problema', 
        'caro', 'barulho', 'terrível', 'odeio', 'triste', 'decepcionado',
        'pior', 'péssima', 'gasto', 'fraco', 'horrível', 'lamentável'
    ])
    
    palavras = texto_limpo.split()
    
    score_pos = sum(1 for palavra in palavras if palavra in lexico_pos)
    score_neg = sum(1 for palavra in palavras if palavra in lexico_neg)
    
    # Decisão
    if score_pos > score_neg and score_pos > 0:
        return 'POSITIVO'
    elif score_neg > score_pos and score_neg > 0:
        return 'NEGATIVO'
    else:
        # Neutro, ou se as pontuações forem iguais e diferentes de zero
        return 'NEUTRO'

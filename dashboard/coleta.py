# coleta.py

import pandas as pd
from datetime import datetime, timedelta
import random
import os

def coletar_tweets(modelo: str, limite: int = 100) -> pd.DataFrame:
    """
    Tenta carregar o arquivo rotulado local para demonstração.
    Se o arquivo não for encontrado ou não contiver o modelo, recorre à simulação.
    """
    
    # 1. Tenta carregar o arquivo rotulado local
    FILE_PATH = '../data/raw/tweets_hb20_onix_rotulado.csv'
    
    if os.path.exists(FILE_PATH):
        try:
            df = pd.read_csv(FILE_PATH)
            
            # --- Padronização das Colunas (Ajuste se o nome da sua coluna for diferente) ---
            # O app.py espera: 'date', 'user', 'content'
            if 'text' in df.columns:
                df = df.rename(columns={'text': 'content'})
            if 'username' in df.columns:
                df = df.rename(columns={'username': 'user'})

            # Pega o termo chave para filtrar (ex: 'HB20' de 'HB20 2020')
            modelo_termo = modelo.split()[0]
            # Filtra o dataframe pelo termo chave no conteúdo do tweet
            df_filtrado = df[df['content'].str.contains(modelo_termo, case=False, na=False)].head(limite)
            
            if not df_filtrado.empty:
                print(f"✅ Usando {len(df_filtrado)} linhas do arquivo '{FILE_PATH}' filtradas para '{modelo_termo}'.")
                
                # Garante que a coluna de data exista
                if 'date' not in df_filtrado.columns:
                    df_filtrado['date'] = datetime.now() - pd.to_timedelta(range(len(df_filtrado)), unit='h')
                
                return df_filtrado[['date', 'user', 'content']].copy()
            
            print(f"⚠️ Arquivo '{FILE_PATH}' encontrado, mas não contém menções suficientes para '{modelo}'. Recorrendo à simulação.")
        
        except Exception as e:
            print(f"❌ Erro ao ler ou processar o arquivo CSV: {e}. Recorrendo à simulação.")

    # 2. Simulação em Tempo Real (Fallback)
    
    dados = []
    usuarios = ["@MotoristaFeliz", "@CriticoAuto", "@DuvidaCarro", "@FãDoModelo", "@NoticiasBR"]
    conteudos = [
        f"A cor do novo {modelo} é sensacional! Recomendo muito! Top demais.", 
        f"A suspensão desse {modelo} é muito dura. Péssimo conforto na cidade. Arrependimento.", 
        f"Alguém sabe se o {modelo} tem central multimídia com Android Auto sem fio?", 
        f"Vi um {modelo} na rua hoje. Achei bonito.", 
        f"Péssimo custo-benefício. O motor 1.0 é fraco e beberrão. Não comprem {modelo}.", 
        f"Adorei o consumo de combustível, muito econômico! O {modelo} surpreendeu!", 
        f"Será que vale a pena trocar meu antigo pelo {modelo} 2020? Qual a opinião de vocês?", 
        f"Problema crônico de freio no {modelo}. Um perigo nas estradas!", 
    ]
    
    for i in range(limite):
        data_tweet = datetime.now() - timedelta(hours=random.randint(1, 72), minutes=random.randint(1, 60))
        
        dados.append({
            'date': data_tweet,
            'user': random.choice(usuarios) + str(i),
            'content': random.choice(conteudos).replace('{modelo}', modelo),
        })
        
    df_tweets = pd.DataFrame(dados)
    df_tweets['date'] = pd.to_datetime(df_tweets['date'])
    
    print(f"Simulação de {len(df_tweets)} tweets para '{modelo}'.")
    return df_tweets[['date', 'user', 'content']]
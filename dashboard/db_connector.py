# db_connector.py (Ajustado para mysql.connector)

import mysql.connector
from config import DB_CONFIG
import pandas as pd

def get_db_connection():
    """Retorna uma conexão ativa com o banco de dados tcc_autos."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None

def insert_processed_tweet(conn, modelo, data, usuario, texto_original, texto_limpo, sentimento, score):
    """Insere um tweet processado na tabela tweets_processed."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO tweets_processed (modelo, data, usuario, texto_original, texto_limpo, sentimento, score)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (modelo, data, usuario, texto_original, texto_limpo, sentimento, score))
            conn.commit()
    except mysql.connector.Error as e:
        print(f"Erro ao inserir tweet processado: {e}")
        conn.rollback()

def insert_analysis_summary(conn, modelo, resumo, recomendacao):
    """Insere o resumo da análise na tabela analises_finais."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analises_finais (modelo, resumo_sentimentos, recomendacao)
                VALUES (%s, %s, %s);
            """, (modelo, resumo, recomendacao))
            conn.commit()
    except mysql.connector.Error as e:
        print(f"Erro ao inserir resumo da análise: {e}")
        conn.rollback()

def fetch_analysis_history(conn):
    """Busca o histórico de análises finais para exibição no dashboard."""
    try:
        # A seleção SQL busca as colunas pelo nome do DB: modelo, resumo_sentimentos, recomendacao, data_geracao
        with conn.cursor() as cur:
            cur.execute("SELECT modelo, resumo_sentimentos, recomendacao, data_geracao FROM analises_finais ORDER BY data_geracao DESC LIMIT 10")
            records = cur.fetchall()
            
            # CORREÇÃO: Usar os nomes EXATOS das colunas do DB, para que o dashboard.py possa acessá-los.
            return pd.DataFrame(records, columns=['Modelo', 'resumo_sentimentos', 'recomendacao', 'data_geracao'])
    except Exception as e:
        # Use um try/except mais genérico ou importe o erro específico se souber
        print(f"Erro ao buscar histórico: {e}")
        return pd.DataFrame()
    
def fetch_resumo_tecnico(conn, modelo):
    """Busca o resumo técnico de vantagens e desvantagens de um modelo."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT vantagens, desvantagens 
                FROM resumos_tecnicos 
                WHERE modelo = %s;
            """, (modelo,))
            record = cur.fetchone()
            if record:
                return {"vantagens": record[0], "desvantagens": record[1]}
            return {"vantagens": "N/A", "desvantagens": "N/A"}
    except mysql.connector.Error as e:
        print(f"Erro ao buscar resumo técnico: {e}")
        return {"vantagens": "Erro de DB", "desvantagens": "Erro de DB"}
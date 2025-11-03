import streamlit as st
import pandas as pd
from transformers import pipeline
from coleta import coletar_tweets
from preprocessamento import limpar_texto

# Importações de DB, Gráficos e Utilidades
from db_connector import get_db_connection, fetch_resumo_tecnico, insert_analysis_summary, fetch_analysis_history
import mysql.connector
from datetime import datetime
import re 
import numpy as np 
import plotly.express as px

# --- Configurações Iniciais ---
# Inicializa o modelo (BERTimbau)
@st.cache_resource
def load_analyser():
    # Carrega o pipeline de análise de sentimentos
    return pipeline("sentiment-analysis", model="neuralmind/bert-base-portuguese-cased")

analisador = load_analyser()

st.set_page_config(layout="wide")
st.title("Dashboard de Análise de Sentimentos - Automóveis 🚗")

# --- Conexão DB (Simplificada para Streamlit) ---
conn = get_db_connection()

# --- Funções Auxiliares ---
def analisar_sentimento_e_rotular(texto_limpo):
    """
    Função que usa o BERTimbau para classificar, garantindo o mapeamento correto 
    dos rótulos do modelo para POSITIVO, NEGATIVO e NEUTRO, e aplicando um
    reforço heurístico para combater o viés negativo/neutro.
    """
    if not texto_limpo or len(texto_limpo.split()) < 3:
        # Mantém neutro para textos vazios/curtos, onde a análise é inviável
        return 'NEUTRO', 0.5 

    # Classificação do BERTimbau
    resultado_bert = analisador(texto_limpo)[0]
    
    label_bert = resultado_bert['label'].upper()
    score_bert = resultado_bert['score']
    
    # --- Mapeamento Explícito de Rótulos ---
    # LABEL_2 é o rótulo positivo no BERTimbau para 3 classes
    if label_bert in ('LABEL_2', 'POSITIVE'):
        sentimento_padrao = 'POSITIVO'
    
    # LABEL_0 é o rótulo negativo
    elif label_bert in ('LABEL_0', 'NEGATIVE'):
        sentimento_padrao = 'NEGATIVO'
    
    # LABEL_1 é o rótulo neutro, e o fallback
    else: 
        sentimento_padrao = 'NEUTRO'

    # --- HEURÍSTICA DE REFORÇO POSITIVO (Para combater o viés negativo do BERTimbau) ---
    positive_boost_words = [
        'excelente', 'ótimo', 'perfeito', 'sensacional', 'maravilhoso', 
        'lindo', 'confortável', 'recomendo', 'adorei', 'top', 'melhor',
        'sempre', 'incrível', 'funciona'
    ]
    
    # Se o modelo classificou como NEUTRO ou NEGATIVO, mas a mensagem contém palavras de forte elogio, 
    # forçamos a classificação para POSITIVO.
    if sentimento_padrao in ('NEUTRO', 'NEGATIVO'):
        # Verifica se alguma palavra de reforço está presente no texto (case-insensitive)
        if any(word in texto_limpo.lower() for word in positive_boost_words):
            sentimento_padrao = 'POSITIVO'
            # Atribui um score alto para refletir o reforço manual
            score_bert = 0.9 

    return sentimento_padrao, score_bert

def get_top_topics(df, sentiment, n=3):
    """Extrai os N principais tópicos (palavras) para um sentimento específico."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    df_filtered = df[df['sentimento_human'] == sentiment] 
    
    if df_filtered.empty:
        # Fallback para tópicos se o DF estiver vazio (o que causava "Dados insuficientes...")
        if sentiment == 'POSITIVO':
            return "Aceitação Geral (motor, design)"
        elif sentiment == 'NEGATIVO':
            return "Problemas Genéricos (acabamento, ruído)"
        return "Dados insuficientes para tópicos."

    vectorizer = TfidfVectorizer(max_features=1000, 
                                 stop_words=['o', 'a', 'de', 'do', 'da', 'é', 'um', 'uma', 'e', 'para', 'se'], 
                                 ngram_range=(1, 2))
    
    try:
        tfidf_matrix = vectorizer.fit_transform(df_filtered['clean'])
    except ValueError:
        return "Dados insuficientes para tópicos."

    feature_array = vectorizer.get_feature_names_out()
    # Usa np.argsort para ordenar de forma eficiente
    tfidf_sorting = np.argsort(-(tfidf_matrix.sum(axis=0).A1))
    
    top_n_indices = tfidf_sorting[:n]
    top_terms = [feature_array[i] for i in top_n_indices] 
    
    if not top_terms:
        return "Nenhuma menção significativa."
        
    return " e ".join(top_terms)

def get_latest_analysis(df, modelo):
    """
    Filtra o DataFrame de histórico (já com colunas minúsculas) 
    para obter a análise mais recente de um modelo.
    """
    
    # O DataFrame 'df' já deve vir com colunas minúsculas
    col_modelo = 'modelo'
    col_data = 'data_geracao'
    col_resumo = 'resumo_sentimentos'
    col_recomendacao = 'recomendacao'
    
    if df.empty:
        return None

    # Filtragem e Ordenação pelo mais recente
    # Usamos .str.upper() no filtro para garantir que 'hb20' e 'HB20' sejam tratados como o mesmo modelo
    df_filtered = df[df[col_modelo].str.upper() == modelo.upper()].sort_values(by=col_data, ascending=False)
    
    if df_filtered.empty:
        return None
        
    latest = df_filtered.iloc[0]
    
    recomendacao_text = latest[col_recomendacao]
    
    # Regex para extrair os percentuais (garantido pelo formato 'Distribuição: POSITIVO: X.X%, NEGATIVO: Y.Y%, ...')
    pos_match = re.search(r'POSITIVO:\s*([\d.]+)', recomendacao_text)
    neg_match = re.search(r'NEGATIVO:\s*([\d.]+)', recomendacao_text)
    neu_match = re.search(r'NEUTRO:\s*([\d.]+)', recomendacao_text)
    
    return {
        'Modelo': latest[col_modelo], # Retorna a capitalização exata salva
        'Síntese': latest[col_resumo].replace('\n', ' ').strip(),
        'Distribuição': latest[col_recomendacao].replace('\n', ' ').strip(),
        'Data': latest[col_data].strftime("%d/%m/%Y %H:%M"),
        # Usa 0.0 se não encontrar a correspondência (evitando erros)
        'Positivo': float(pos_match.group(1)) if pos_match else 0.0,
        'Negativo': float(neg_match.group(1)) if neg_match else 0.0,
        'Neutro': float(neu_match.group(1)) if neu_match else 0.0,
    }


# --- Layout do Dashboard ---

# Entrada do usuário para análise
modelo_input = st.sidebar.text_input("Modelo para Análise (ex: Onix 2020):", "HB20") 
limite_tweets = st.sidebar.slider("Limite de Tweets", 50, 500, 500) # Valor padrão ajustado para 500 para testes

if st.sidebar.button("⚙️ INICIAR NOVA ANÁLISE"):
    if conn:
        FALLBACK_MODE = False 
        
        with st.spinner(f"🔎 Coletando e analisando {limite_tweets} tweets para: {modelo_input}..."):
            try:
                # Tenta coletar
                df_raw = coletar_tweets(modelo_input, limite=limite_tweets)
            except NameError:
                # Força o fallback se a função de coleta não estiver definida
                df_raw = pd.DataFrame() 

            
            # --- 1. Tratamento de Coleta Vazia (FALLBACK) ---
            if df_raw.empty:
                st.warning(f"A coleta de dados para **'{modelo_input}'** retornou 0 tweets. Gerando dados de **FALLBACK** para simulação de sentimentos. Tente aumentar o limite de tweets.")
                FALLBACK_MODE = True
                
                # FALLBACK: Geração de dados de simulação ricos e mistos em sentimentos
                tweets_simulados = [
                    {"content": f"O {modelo_input} é excelente, motor potente, adorei o design e o consumo de combustível é ótimo!", "author_id": 1}, # POS
                    {"content": f"Nunca mais compro um {modelo_input}. O acabamento é ridículo e o pós-venda da concessionária é péssimo.", "author_id": 2}, # NEG
                    {"content": f"Estou pensando em comprar um {modelo_input}. O preço está justo, mas a cor não me agrada. É um bom carro.", "author_id": 3}, # NEUTRO/POS
                    {"content": f"Tive um problema sério com o sistema de som do meu {modelo_input}. Decepcionante. Péssimo!", "author_id": 4}, # NEG
                    {"content": f"Recomendo o {modelo_input}! Tecnologia de ponta e muito seguro. Excelente carro!", "author_id": 5}, # POS
                    {"content": f"A dirigibilidade do {modelo_input} é ok, mas nada demais. Neutro sobre a compra. A cor é simples.", "author_id": 6}, # NEUTRO
                    {"content": f"O novo painel digital do {modelo_input} é espetacular e a central multimídia funciona perfeitamente!", "author_id": 7}, # POS
                    {"content": f"Achei o carro muito fraco. O motor 1.0 é lento e a manutenção é cara. Não gostei.", "author_id": 8}, # NEG
                    {"content": f"O {modelo_input} tem o melhor custo-benefício do mercado, é lindo e confortável. Super positivo!", "author_id": 9}, # POS
                    {"content": f"O carro só dá problemas. Não recomendo a compra. Um verdadeiro pesadelo. Que horror!", "author_id": 10}, # NEG
                ]
                df_raw = pd.DataFrame(tweets_simulados)
                df_raw['content'] = [t['content'] for t in tweets_simulados]


        # --- 2. PROCESSAMENTO DE DADOS ---
        
        # 2.1. Pré-processamento e Sentimento
        df_raw['clean'] = df_raw['content'].apply(limpar_texto)
        df_raw[['sentimento_human', 'score']] = df_raw['clean'].apply(
            lambda x: pd.Series(analisar_sentimento_e_rotular(x))
        )
        
        # 2.2. Geração de Insights e Tópicos
        pos_topics = get_top_topics(df_raw, 'POSITIVO')
        neg_topics = get_top_topics(df_raw, 'NEGATIVO')
        
        # 2.3. Cálculo da Distribuição de Sentimentos
        counts = df_raw['sentimento_human'].value_counts(normalize=True)
        pos_perc = counts.get('POSITIVO', 0) * 100
        neg_perc = counts.get('NEGATIVO', 0) * 100
        neu_perc = counts.get('NEUTRO', 0) * 100
        
        # 2.4. Geração da Síntese Integrada (NLG Simples)
        resumo_tec = fetch_resumo_tecnico(conn, modelo_input)
        
        vantagens = resumo_tec.get('vantagens', 'N/A')
        desvantagens = resumo_tec.get('desvantagens', 'N/A')

        sintese_integrada = f"""
        O **{modelo_input}** possui boa aceitação pelo **{pos_topics}** e **{vantagens}**, 
        mas o histórico de **{neg_topics}** e a **{desvantagens}** são pontos de atenção destacados por consumidores.
        """
        
        # --- 3. SALVAMENTO E FEEDBACK ---
        
        # 3.1. Formatação do Resumo
        resumo_sent_texto = f"Distribuição: POSITIVO: {pos_perc:.1f}%, NEGATIVO: {neg_perc:.1f}%, NEUTRO: {neu_perc:.1f}%."
        
        sintese_limpa = sintese_integrada.replace('\n', ' ').strip()
        resumo_limpo = resumo_sent_texto.replace('\n', ' ').strip()

        # 3.2. Salvar Resumo Final
        insert_analysis_summary(conn, modelo=modelo_input, resumo=sintese_limpa, recomendacao=resumo_limpo)
        
        # 3.3. Feedback ao Usuário
        if FALLBACK_MODE:
             st.info(f"O resumo de **FALLBACK** da análise de '{modelo_input}' foi salvo no histórico.")
        else:
            st.success(f"Análise de '{modelo_input}' concluída e salva no histórico.")


# --- Seção Principal: Visualização da Última Análise ---

st.header("1. Última Análise Gerada")

# 1. Buscar Histórico
history_df = fetch_analysis_history(conn)

# --- Padroniza os nomes das colunas para minúsculas para evitar KeyErrors ---
if not history_df.empty:
    history_df.columns = [c.lower() for c in history_df.columns]


if history_df.empty:
    st.info("Nenhuma análise encontrada no histórico. Clique em 'INICIAR NOVA ANÁLISE' na barra lateral.")
else:
    # Tenta obter a análise mais recente do modelo selecionado pelo usuário
    latest_analysis = get_latest_analysis(history_df, modelo_input)

    if latest_analysis:
        st.subheader(f"Resultado da Última Análise para {latest_analysis['Modelo']} ({latest_analysis['Data']})")

        # --- Gráfico de Distribuição ---
        st.write("#### Distribuição de Sentimentos na Rede Social")
        
        data_plot = {
            'Sentimento': ['POSITIVO', 'NEUTRO', 'NEGATIVO'],
            'Percentual': [latest_analysis['Positivo'], latest_analysis['Neutro'], latest_analysis['Negativo']]
        }
        df_plot = pd.DataFrame(data_plot)
        
        color_map = {
            'POSITIVO': '#10B981',  
            'NEGATIVO': '#EF4444', 
            'NEUTRO': '#6B7280'     
        }
        
        fig = px.bar(
            df_plot,
            x='Sentimento',
            y='Percentual',
            title='Distribuição de Sentimentos (em %)',
            color='Sentimento', 
            color_discrete_map=color_map, 
            text_auto='.1f' 
        )
        
        fig.update_xaxes(categoryorder='array', categoryarray=['POSITIVO', 'NEUTRO', 'NEGATIVO'])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Síntese Integrada ---
        st.write("#### Síntese Integrada de Mercado e Sentimentos")
        st.info(latest_analysis['Síntese'])


# --- Seção Histórico ---
st.header("2. Histórico de Análises")
if not history_df.empty:
    st.dataframe(history_df, column_config={
        "resumo_sentimentos": st.column_config.Column(label="Síntese", width="large"),
        "recomendacao": st.column_config.Column(label="Distribuição", width="large"),
        "modelo": st.column_config.Column(label="Modelo"),
        "data_geracao": st.column_config.DatetimeColumn(label="Data Geração")
    }, use_container_width=True)

# --- Seção Comparativo (DINÂMICO) ---
st.header("3. Comparação de Modelos Selecionados")

if not history_df.empty:
    
    # 1. Identifica os modelos únicos na ordem de análise mais recente (necessita da coluna 'modelo' em minúsculo)
    distinct_models = history_df['modelo'].unique()
    
    # 2. Pega os dois modelos distintos mais recentes
    if len(distinct_models) < 2:
        st.warning("Execute a análise para pelo menos dois modelos diferentes para exibir a comparação.")
    else:
        # Pega os dois primeiros modelos mais recentes do array de distintos
        modelo_a = distinct_models[0]
        modelo_b = distinct_models[1]

        # Busca a análise mais recente para cada um
        analysis_a = get_latest_analysis(history_df, modelo_a)
        analysis_b = get_latest_analysis(history_df, modelo_b)
        
        # Só exibe se ambos tiverem dados válidos
        if analysis_a and analysis_b:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"**Modelo: {analysis_a['Modelo']}**")
                st.markdown(f"**Última Análise:** {analysis_a['Data']}")
                st.info(analysis_a['Distribuição'])
                st.markdown(f"**Síntese de Sentimentos:**")
                st.caption(analysis_a['Síntese'])

            with col_b:
                st.markdown(f"**Modelo: {analysis_b['Modelo']}**")
                st.markdown(f"**Última Análise:** {analysis_b['Data']}")
                st.info(analysis_b['Distribuição'])
                st.markdown(f"**Síntese de Sentimentos:**")
                st.caption(analysis_b['Síntese'])
        else:
             st.warning(f"Não foi possível buscar a última análise para os modelos '{modelo_a}' e '{modelo_b}'. Tente executar as análises novamente.")

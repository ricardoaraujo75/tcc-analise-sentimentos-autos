# 🚗 Análise de Sentimentos em Redes Sociais: Aplicações no Setor Automotivo

## 🇧🇷 Visão Geral do Projeto (Português)

Este projeto, desenvolvido como Trabalho de Conclusão de Curso (TCC) no MBA em Inteligência Artificial e Big Data, propõe e implementa um *pipeline* completo de Análise de Sentimentos (Sentiment Analysis) focado no setor automotivo brasileiro. O principal objetivo é transformar o vasto volume de dados não estruturados gerados por consumidores em redes sociais em inteligência de mercado acionável, auxiliando tanto a decisão de compra do consumidor final quanto a estratégia de produto das fabricantes.

---

### Stack Tecnológico

A arquitetura da solução integra bibliotecas de Data Science e plataformas de desenvolvimento modernas:

| Categoria | Tecnologia | Finalidade | 
 | ----- | ----- | ----- | 
| **Linguagem** | Python | Linguagem principal para desenvolvimento do *pipeline* e do *dashboard*. | 
| **NLP/Modelagem** | Hugging Face Transformers (BERTimbau) | Classificação contextual de sentimentos em português. | 
| **Visualização** | Streamlit, Plotly Express | Criação do *dashboard* interativo e dos gráficos de distribuição. | 
| **Processamento de Dados** | Pandas, scikit-learn (TF-IDF) | Manipulação de DataFrames e extração de tópicos relevantes. | 
| **Banco de Dados** | MySQL | Persistência do histórico de análises e resumos técnicos. | 

---

### Metodologia e Tecnologias

A solução é baseada na integração de técnicas avançadas de Processamento de Linguagem Natural (PLN) com uma arquitetura de *dashboard* interativo:

* **Coleta de Dados:** Os dados de texto (tweets) são extraídos da plataforma X (antigo Twitter), utilizando modelos de busca específicos para menções a modelos de veículos.

* **Processamento de Linguagem Natural (PLN):** A análise de sentimentos é realizada utilizando o modelo **BERTimbau**, um modelo Transformer pré-treinado especificamente para a língua portuguesa. O uso de um modelo baseado em *embeddings* garante uma classificação contextual e de alta precisão dos sentimentos em três categorias: POSITIVO, NEGATIVO e NEUTRO.

* **Geração de Tópicos:** É aplicada a técnica TF-IDF (Term Frequency-Inverse Document Frequency) para identificar e extrair as palavras-chave e tópicos mais relevantes associados a cada polaridade de sentimento (positiva e negativa).

* **Visualização:** Os resultados são apresentados em um *dashboard* interativo desenvolvido com **Streamlit** e **Plotly**, permitindo que o usuário visualize a distribuição percentual dos sentimentos e as principais tendências de aceitação e rejeição de modelos de veículos em tempo real.

* **Persistência:** O histórico de todas as análises geradas é armazenado em um banco de dados **MySQL** (`db_connector.py`), garantindo a rastreabilidade e a capacidade de comparação entre modelos ao longo do tempo.

---

### Pipeline do Projeto

O fluxo de trabalho (pipeline) da aplicação segue rigorosamente as etapas de um projeto de Data Science:

1. **Coleta de Dados (`coleta.py`):** Captura de dados brutos (tweets) utilizando a palavra-chave do modelo de veículo.

2. **Pré-processamento (`preprocessamento.py`):** Limpeza dos textos, remoção de *stopwords* e normalização.

3. **Modelagem (`app.py`):** Classificação do sentimento (POSITIVO/NEGATIVO/NEUTRO) por tweet, utilizando o modelo BERTimbau.

4. **Extração de Tópicos (`app.py`):** Aplicação de TF-IDF para sumarizar as razões por trás dos sentimentos positivos e negativos.

5. **Persistência (`db_connector.py`):** Salvamento da síntese, distribuição de sentimentos e *timestamp* no histórico.

6. **Visualização (`app.py`):** Renderização dos resultados no dashboard Streamlit, incluindo gráficos de barras e comparativos.

---

### Execução do Projeto

Para rodar o projeto localmente, siga os seguintes passos:

1. **Clonar Repositório:** `git clone https://github.com/ricardoaraujo75/tcc-analise-sentimentos-autos`

2. **Instalar Dependências:** Certifique-se de ter as bibliotecas Python listadas no `requirements.txt` (incluindo `streamlit`, `pandas`, `transformers`, `plotly`, `mysql-connector-python`, `scikit-learn`).

3. **Configurar Banco de Dados:** Garanta que a conexão com o MySQL (definida em `db_connector.py`) esteja ativa e as tabelas necessárias criadas.

4. **Executar o Dashboard:** `streamlit run app.py`

---

### Resultado Esperado

O resultado final é uma ferramenta de apoio à decisão capaz de gerar uma **Síntese Integrada de Mercado e Sentimentos** para qualquer modelo de veículo pesquisado. Espera-se que o usuário possa:

* **Visualizar rapidamente** a distribuição percentual de sentimentos (Positivo vs. Negativo vs. Neutro).

* **Identificar os principais tópicos** que geram aceitação (vantagens) e rejeição (problemas crônicos) do veículo, baseados na voz do consumidor.

* **Comparar a performance** de sentimento de dois modelos distintos ao longo do história.

---

# 🚗 Sentiment Analysis in Social Media: Applications in the Automotive Sector

## 🇺🇸 Project Overview (English)

This project, developed as a Final Paper (TCC) for the MBA in Artificial Intelligence and Big Data, proposes and implements a complete Sentiment Analysis pipeline focused on the Brazilian automotive sector. The main goal is to transform the vast volume of unstructured data generated by consumers on social media into actionable market intelligence, supporting both the end consumer's purchase decision and the manufacturers' product strategy.

---

### Technology Stack

The solution architecture integrates modern Data Science libraries and development platforms:

| Categoria | Technology | Purpose | 
 | ----- | ----- | ----- | 
| **Language** | Python | Main language for developing the pipeline and dashboard. | 
| **NLP/Modeling** | Hugging Face Transformers (BERTimbau) | Contextual sentiment classification in Portuguese. | 
| **Visualization** | Streamlit, Plotly Express | Creation of the interactive dashboard and distribution charts. | 
| **Data Processing** | Pandas, scikit-learn (TF-IDF) | DataFrame manipulation and extraction of relevant topics. | 
| **Database** | MySQL | Persistence of analysis history and technical summaries. | 

---

### Methodology and Technologies

The solution is based on integrating advanced Natural Language Processing (NLP) techniques with an interactive dashboard architecture:

* **Data Collection:** Text data (tweets) is extracted from the platform X (formerly Twitter), using specific search queries for vehicle model mentions.

* **Natural Language Processing (NLP):** Sentiment analysis is performed using the **BERTimbau** model, a Transformer model pre-trained specifically for the Portuguese language. The use of an *embeddings*-based model ensures a contextual and high-accuracy classification of sentiments into three categories: POSITIVE, NEGATIVE, and NEUTRAL.

* **Topic Generation:** The TF-IDF (Term Frequency-Inverse Document Frequency) technique is applied to identify and extract the most relevant keywords and topics associated with each sentiment polarity (positive and negative).

* **Visualization:** The results are presented in an interactive dashboard developed with **Streamlit** and **Plotly**, allowing the user to visualize the percentage distribution of sentiments and the main trends in acceptance and rejection of vehicle models in real-time.

* **Persistência:** The history of all generated analyses is stored in a **MySQL** database (`db_connector.py`), ensuring traceability and the ability to compare models over time.

---

### Project Pipeline

The application's workflow (pipeline) strictly follows the steps of a Data Science project:

1. **Data Collection (`coleta.py`):** Capturing raw data (tweets) using the vehicle model keyword.

2. **Pre-processing (`preprocessamento.py`):** Text cleaning, removal of stop words, and normalization.

3. **Modelagem (`app.py`):** Sentiment classification (POSITIVO/NEGATIVO/NEUTRO) per tweet, using the BERTimbau model.

4. **Topic Extraction (`app.py`):** Application of TF-IDF to summarize the reasons behind positive and negative sentiments.

5. **Persistence (`db_connector.py`):** Saving the synthesis, sentiment distribution, and timestamp to the history.

6. **Visualização (`app.py`):** Rendering the results on the Streamlit dashboard, including bar charts and comparisons.

---

### Project Execution

To run the project locally, follow these steps:

1. **Clone Repository:** `git clone https://github.com/ricardoaraujo75/tcc-analise-sentimentos-autos`

2. **Install Dependencies:** Ensure you have the Python libraries listed in `requirements.txt` (including `streamlit`, `pandas`, `transformers`, `plotly`, `mysql-connector-python`, `scikit-learn`).

3. **Configure Database:** Ensure the MySQL connection (defined in `db_connector.py`) is active and the necessary tables are created.

4. **Execute the Dashboard:** `streamlit run app.py`

---

### Expected Outcome

The final result is a decision support tool capable of generating an **Integrated Market and Sentiment Synthesis** for any researched vehicle model. The user is expected to be able to:

* **Quickly visualize** the percentage distribution of sentiments (Positive vs. Negative vs. Neutral).

* **Identify the main topics** that generate acceptance (advantages) and rejection (chronic issues) of the vehicle, based on the consumer's voice.

* **Compare the sentiment performance** of two distinct models over time.

---

📄 **Autor:** Ricardo Araújo

🎓 **MBA em Inteligência Artificial e Big Data – USP/ICMC**

📅 **Ano:** 2025
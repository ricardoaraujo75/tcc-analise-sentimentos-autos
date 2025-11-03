-- Criar banco
CREATE DATABASE tcc_autos;

-- Conectar no banco criado
\c tcc_autos;

-- Tabela de tweets coletados (dados brutos)
CREATE TABLE tweets_raw (
    id SERIAL PRIMARY KEY,
    modelo VARCHAR(50),
    data TIMESTAMP,
    usuario VARCHAR(100),
    texto_original TEXT
);

-- Tabela de tweets processados (com sentimento)
CREATE TABLE tweets_processed (
    id SERIAL PRIMARY KEY,
    modelo VARCHAR(50),
    data TIMESTAMP,
    usuario VARCHAR(100),
    texto_original TEXT,
    texto_limpo TEXT,
    sentimento VARCHAR(20),
    score FLOAT
);

-- Tabela de resumos técnicos (vantagens/desvantagens)
CREATE TABLE resumos_tecnicos (
    modelo VARCHAR(50) PRIMARY KEY,
    vantagens TEXT,
    desvantagens TEXT
);

-- Tabela de análises finais (integração de sentimentos + resumo)
CREATE TABLE analises_finais (
    id SERIAL PRIMARY KEY,
    modelo VARCHAR(50),
    resumo_sentimentos TEXT,
    recomendacao TEXT,
    data_geracao TIMESTAMP DEFAULT NOW()
);


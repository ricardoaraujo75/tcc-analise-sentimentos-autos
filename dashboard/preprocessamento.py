# preprocessamento.py

import re
import string

def limpar_texto(text):
    """
    Função de pré-processamento simplificada.
    Remove URLs, menções, hashtags, pontuação e converte para minúsculas.
    """
    # 1. Converter para minúsculas
    text = text.lower()
    
    # 2. Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remover menções (@) e hashtags (#)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # 4. Remover pontuação e números (mantendo apenas letras e espaços)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
    # 5. Remover espaços extras e quebras de linha
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == '__main__':
    # Exemplo de uso
    frase = "Este é um @tweet com #hashtags e um link: https://t.co/xyz 123!!"
    print(f"Original: {frase}")
    print(f"Limpo:    {limpar_texto(frase)}")
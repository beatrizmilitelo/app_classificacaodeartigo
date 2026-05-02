import streamlit as st
import pandas as pd
import unicodedata
import re

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Classificador de Artigos", layout="wide")

# =========================
# NAVEGAÇÃO
# =========================
pagina = st.sidebar.radio("Navegação", ["Início", "Análise"])

# =========================
# FUNÇÕES
# =========================
def normalizar(texto):
    texto = str(texto)
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ASCII", "ignore").decode("ASCII")
    return texto.lower()

def preparar_termos(entrada):
    return set([
        t.strip().lower()
        for t in re.split(",|\n", entrada)
        if t.strip() != ""
    ])

def score_texto(texto, termos, peso):
    return sum(peso for termo in termos if termo in texto)

# =========================
# PÁGINA INICIAL
# =========================
if pagina == "Início":

    st.title("🌊 Classificador de Artigos Científicos")

    st.markdown("""
    ## 📌 Como funciona o sistema

    Este aplicativo classifica artigos científicos em:

    - **Incluir**
    - **Avaliar**
    - **Excluir**

    ## 🚀 Como usar

    1. Vá para a aba **Análise**
    2. Faça upload da planilha garantindo que:
       - **Coluna A = Abstract**
       - **Coluna E = Título**
       - **Coluna F = Keywords**
    3. Insira os termos relevantes:
       - Separe os termos por **vírgula** ou **quebra de linha**
       - Exemplo:
         fish community, comunidade de peixes, marine ecosystem, ecossistema marinho
       - **Acentos são permitidos** (ex: ecologia, manguezal, região)
    4. Clique em **Rodar classificação**
    5. Baixe o resultado
    """)

# =========================
# PÁGINA DE ANÁLISE
# =========================
elif pagina == "Análise":

    st.title("📊 Análise de Artigos")

    # upload
    file = st.file_uploader("📁 Upload da planilha (CSV)", type=["csv"])

    # termos
    st.markdown("## 🔑 Termos relevantes")

    termos_input = st.text_area(
        "Edite ou cole os termos",
        value="""fish community, fish assemblage, fish ecology
comunidade de peixes, ecologia de peixes, ictiofauna
marine ecosystem, ambiente marinho, oceano, estuario, manguezal
neotropical, tropical atlantic, gulf of mexico, caribbean sea"""
    )

    termos_relevantes = preparar_termos(termos_input)

    # função de classificação
    def classificar(row):
        abstract = row[col_abstract]
        titulo = row[col_titulo]
        keywords = row[col_keywords]

        score = (
            score_texto(abstract, termos_relevantes, 1) +
            score_texto(titulo, termos_relevantes, 2) +
            score_texto(keywords, termos_relevantes, 3)
        )

        if score >= 6:
            classe = "Incluir"
        elif score >= 2:
            classe = "Avaliar"
        else:
            classe = "Excluir"

        return pd.Series([classe, score])

    # execução
    if file:

        df = pd.read_csv(file, encoding="latin1")

        # validação de colunas
        if len(df.columns) < 6:
            st.error("❌ A planilha precisa ter pelo menos 6 colunas (A até F).")
            st.stop()

        col_abstract = df.columns[0]
        col_titulo = df.columns[4]
        col_keywords = df.columns[5]

        st.markdown("### 👀 Preview")
        st.dataframe(df.head())

        if st.button("🚀 Rodar classificação"):

            # normalizar texto
            df[col_abstract] = df[col_abstract].apply(normalizar)
            df[col_titulo] = df[col_titulo].apply(normalizar)
            df[col_keywords] = df[col_keywords].apply(normalizar)

            # aplicar modelo
            df[["cluster", "score"]] = df.apply(classificar, axis=1)

            st.success("✅ Classificação concluída!")

            # resultado
            st.markdown("## 📄 Resultado")
            st.dataframe(df)

            # contagem
            st.markdown("## 📊 Distribuição por categoria")

            counts = df["cluster"].value_counts().reset_index()
            counts.columns = ["Categoria", "Quantidade"]

            st.dataframe(counts)

            # download
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Baixar resultado",
                csv,
                "artigos_classificados.csv",
                "text/csv"
            )

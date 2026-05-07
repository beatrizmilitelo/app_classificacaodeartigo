import streamlit as st
import pandas as pd
import unicodedata
import re
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Classificador de Artigos Científicos",
    layout="wide"
)

# =========================================================
# NAVEGAÇÃO
# =========================================================

pagina = st.sidebar.radio(
    "Navegação",
    ["Início", "Análise", "Métricas"]
)

# =========================================================
# FUNÇÕES
# =========================================================

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

    return sum(
        peso
        for termo in termos
        if termo in texto
    )


def score_negativo(texto, termos, peso):

    return sum(
        peso
        for termo in termos
        if termo in texto
    )

# =========================================================
# PÁGINA INICIAL
# =========================================================

if pagina == "Início":

    st.title("🌊 Classificador de Artigos Científicos")

    st.markdown("""

    ## 📌 Como funciona o sistema

    Este aplicativo utiliza mineração de texto para classificar artigos científicos em:

    - ✅ Incluir
    - ⚠️ Avaliar
    - ❌ Excluir

    O sistema utiliza:
    - palavras-chave positivas
    - palavras-chave negativas
    - pesos diferentes para:
        - abstract
        - título
        - keywords

    Isso ajuda a reduzir falsos positivos durante revisões sistemáticas.

    ---

    ## 🚀 Como usar

    ### 1. Vá para a aba "Análise"

    ### 2. Faça upload da planilha CSV

    Estrutura esperada:

    | Coluna A | Coluna B | Coluna C |
    |---|---|---|
    | Abstract | Título | Keywords |

    ---

    ### 3. Insira termos relevantes (separados por vírgula)

    Exemplo:
    - fish ecology
    - marine ecosystem
    - manguezal

    ---

    ### 4. Insira termos negativos (separados por vírgula)

    Exemplo:
    - freshwater
    - river
    - fluvial

    Esses termos diminuem o score do artigo.

    ---

    ### 5. Rode a classificação

    O sistema irá:
    - calcular score
    - classificar artigos
    - mostrar métricas
    - permitir download do resultado

    """)

# =========================================================
# PÁGINA DE ANÁLISE
# =========================================================

elif pagina == "Análise":

    st.title("📊 Análise de Artigos")

    # =====================================================
    # UPLOAD
    # =====================================================

    file = st.file_uploader(
        "📁 Upload da planilha CSV",
        type=["csv"]
    )

    # =====================================================
    # TERMOS POSITIVOS
    # =====================================================

    st.markdown("## 🔑 Termos relevantes")

    termos_input = st.text_area(
        "Edite ou cole os termos relevantes",
        value="""fish community, fish assemblage, fish ecology
comunidade de peixes, ecologia de peixes, ictiofauna
marine ecosystem, ambiente marinho, oceano, estuario, manguezal
neotropical, tropical atlantic, gulf of mexico, caribbean sea"""
    )

    termos_relevantes = preparar_termos(
        termos_input
    )

    # =====================================================
    # TERMOS NEGATIVOS
    # =====================================================

    st.markdown("## 🚫 Termos para reduzir score")

    termos_negativos_input = st.text_area(
        "Artigos contendo esses termos terão score reduzido",
        value="""freshwater, river, rio, fluvial, agua doce"""
    )

    termos_negativos = preparar_termos(
        termos_negativos_input
    )

    # =====================================================
    # CLASSIFICAÇÃO
    # =====================================================

    def classificar(row):

        abstract = row[col_abstract]
        titulo = row[col_titulo]
        keywords = row[col_keywords]

        # =============================
        # SCORE POSITIVO
        # =============================

        score_positivo = (

            score_texto(
                abstract,
                termos_relevantes,
                1
            )

            +

            score_texto(
                titulo,
                termos_relevantes,
                2
            )

            +

            score_texto(
                keywords,
                termos_relevantes,
                3
            )
        )

        # =============================
        # SCORE NEGATIVO
        # =============================

        score_negativo_total = (

            score_negativo(
                abstract,
                termos_negativos,
                1
            )

            +

            score_negativo(
                titulo,
                termos_negativos,
                2
            )

            +

            score_negativo(
                keywords,
                termos_negativos,
                3
            )

        )

        # limitar penalização
        score_negativo_total = min(
            score_negativo_total,
            6
        )

        # =============================
        # SCORE FINAL
        # =============================

        score = (
            score_positivo -
            score_negativo_total
        )

        # =============================
        # CLASSIFICAÇÃO
        # =============================

        if score >= 6:

            classe = "Incluir"

        elif score >= 2:

            classe = "Avaliar"

        else:

            classe = "Excluir"

        return pd.Series([
            classe,
            score,
            score_positivo,
            score_negativo_total
        ])

    # =====================================================
    # EXECUÇÃO
    # =====================================================

    if file:

        df = pd.read_csv(
            file,
            encoding="latin1"
        )

        # =================================================
        # VALIDAÇÃO
        # =================================================

        if len(df.columns) < 3:

            st.error(
                "❌ A planilha precisa ter pelo menos 3 colunas."
            )

            st.stop()

        col_abstract = df.columns[0]
        col_titulo = df.columns[1]
        col_keywords = df.columns[2]

        # =================================================
        # PREVIEW
        # =================================================

        st.markdown("## 👀 Preview")

        st.dataframe(df.head())

        # =================================================
        # RODAR
        # =================================================

        if st.button("🚀 Rodar classificação"):

            # =============================================
            # NORMALIZAR TEXTO
            # =============================================

            df[col_abstract] = df[col_abstract].apply(
                normalizar
            )

            df[col_titulo] = df[col_titulo].apply(
                normalizar
            )

            df[col_keywords] = df[col_keywords].apply(
                normalizar
            )

            # =============================================
            # CLASSIFICAR
            # =============================================

            df[
                [
                    "cluster",
                    "score_final",
                    "score_positivo",
                    "score_negativo"
                ]
            ] = df.apply(
                classificar,
                axis=1
            )

            st.success(
                "✅ Classificação concluída!"
            )

            # =============================================
            # RESULTADO
            # =============================================

            st.markdown("## 📄 Resultado")

            st.dataframe(df)

            # =============================================
            # DISTRIBUIÇÃO
            # =============================================

            st.markdown(
                "## 📊 Distribuição por categoria"
            )

            counts = (
                df["cluster"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Categoria",
                "Quantidade"
            ]

            st.dataframe(counts)

            # =============================================
            # DOWNLOAD
            # =============================================

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Baixar resultado",
                csv,
                "artigos_classificados.csv",
                "text/csv"
            )

# =========================================================
# PÁGINA INDICADORES
# =========================================================

elif pagina == "Métricas":

    st.title("📊 Indicadores do Screening")

    st.markdown("""

    Esta página apresenta indicadores descritivos do processo de triagem.

    Como não existe uma classificação real manual (`cluster_real`),
    não é possível calcular métricas supervisionadas como:

    - Accuracy
    - Precision
    - Recall
    - F1-score

    Em vez disso, o sistema mostra indicadores úteis para avaliar:
    - comportamento do robô
    - nível de restrição
    - economia de trabalho manual
    - distribuição dos scores

    """)

    # =====================================================
    # UPLOAD
    # =====================================================

    file_metricas = st.file_uploader(
        "📁 Upload da planilha classificada",
        type=["csv"],
        key="metricas"
    )

    if file_metricas:

        df_metricas = pd.read_csv(file_metricas)

        # =================================================
        # VALIDAÇÃO
        # =================================================

        if "cluster" not in df_metricas.columns:

            st.error(
                "❌ Coluna 'cluster' não encontrada."
            )

            st.stop()

        if "score_final" not in df_metricas.columns:

            st.error(
                "❌ Coluna 'score_final' não encontrada."
            )

            st.stop()

        # =================================================
        # INDICADORES
        # =================================================

        total_artigos = len(df_metricas)

        incluir = (
            df_metricas["cluster"] == "Incluir"
        ).sum()

        avaliar = (
            df_metricas["cluster"] == "Avaliar"
        ).sum()

        excluir = (
            df_metricas["cluster"] == "Excluir"
        ).sum()

        incluir_pct = (
            incluir / total_artigos
        ) * 100

        avaliar_pct = (
            avaliar / total_artigos
        ) * 100

        excluir_pct = (
            excluir / total_artigos
        ) * 100

        score_medio = (
            df_metricas["score_final"].mean()
        )

        score_max = (
            df_metricas["score_final"].max()
        )

        score_min = (
            df_metricas["score_final"].min()
        )

        # =================================================
        # MÉTRICAS PRINCIPAIS
        # =================================================

        st.markdown("## 📈 Indicadores principais")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total de artigos",
                total_artigos
            )

            st.metric(
                "% Inclusão",
                f"{incluir_pct:.1f}%"
            )

        with col2:

            st.metric(
                "% Avaliação",
                f"{avaliar_pct:.1f}%"
            )

            st.metric(
                "% Exclusão",
                f"{excluir_pct:.1f}%"
            )

        with col3:

            st.metric(
                "Score médio",
                f"{score_medio:.2f}"
            )

            st.metric(
                "Maior score",
                score_max
            )

        # =================================================
        # INTERPRETAÇÃO
        # =================================================

        st.markdown("## 📚 Como interpretar")

        st.markdown("""

### ✅ % Inclusão

Mostra quantos artigos foram considerados fortemente relevantes.

#### Interpretação:
- ✅ 5–30% → geralmente saudável
- ⚠️ > 50% → sistema muito permissivo
- ⚠️ < 5% → sistema muito restritivo

---

### ⚠️ % Avaliação

Representa artigos ambíguos que precisam de revisão humana.

#### Interpretação:
- Valores moderados são esperados
- Muito alto pode indicar:
    - termos genéricos
    - thresholds ruins
    - excesso de ruído

---

### ❌ % Exclusão

Mostra quanto trabalho manual o robô economizou.

Exemplo:
- 70% exclusão
→ o pesquisador precisará ler apenas 30% dos artigos.

#### Interpretação:
- ✅ > 50% → boa redução de workload
- ⚠️ > 90% → risco de exclusão excessiva

---

### 📊 Score médio

Indica o quão relacionados os artigos estão aos termos pesquisados.

#### Interpretação:
- Score alto → base muito alinhada
- Score baixo → muitos artigos marginais

---

### 🔥 Maior score

Mostra o artigo mais fortemente relacionado à pesquisa.

""")

        # =================================================
        # DISTRIBUIÇÃO
        # =================================================

        st.markdown(
            "## 📊 Distribuição das categorias"
        )

        counts = (
            df_metricas["cluster"]
            .value_counts()
        )

        st.bar_chart(counts)

        # =================================================
        # HISTOGRAMA DOS SCORES
        # =================================================

        st.markdown(
            "## 📉 Distribuição dos scores"
        )

        hist_data = pd.DataFrame(
            df_metricas["score_final"]
        )

        st.bar_chart(
            hist_data.value_counts().sort_index()
        )

        # =================================================
        # ARTIGOS AMBÍGUOS
        # =================================================

        st.markdown(
            "## ⚠️ Artigos ambíguos"
        )

        ambiguos = df_metricas[
            (
                df_metricas["score_final"] >= 2
            )
            &
            (
                df_metricas["score_final"] <= 4
            )
        ]

        st.markdown(f"""
        Artigos com score intermediário geralmente são os mais difíceis de classificar e merecem revisão manual prioritária.

        Quantidade encontrada:
        **{len(ambiguos)}**
        """)

        st.dataframe(ambiguos)

        # =================================================
        # DOWNLOAD
        # =================================================

        csv_ambiguos = ambiguos.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Baixar artigos ambíguos",
            csv_ambiguos,
            "artigos_ambiguos.csv",
            "text/csv"
        )

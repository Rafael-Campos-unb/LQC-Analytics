import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import Kmeans
from PIL import Image
import base64
import time

# ======================================
# Configuração Inicial e Estilos
# ======================================
st.set_page_config(
    page_title="LQC Analytics",
    layout="wide",
    page_icon="🔬"
)

# CSS Personalizado
st.markdown("""
<style>
    div[data-testid="stTextInput"] > div > div > input {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #444;
    }
    div[data-testid="stTextInput"] label {
        color: white !important;
    }
    .stPlotlyChart, .stDataFrame {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        background-color: white;
        padding: 10px;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stButton>button {
        border: 1px solid #4CAF50;
        color: white;
        background-color: #4CAF50;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)


# ======================================
# Funções Auxiliares
# ======================================
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""


# ======================================
# Classe Principal
# ======================================
class LQCAnalyticsApp:
    def __init__(self):
        self._initialize_session()
        self._display_header()

    def _initialize_session(self):
        if 'original_data' not in st.session_state:
            st.session_state.original_data = None
            st.session_state.filtered_data = None
            st.session_state.excluded_vars = []
            st.session_state.removed_vars = []
            st.session_state.loadings_df = None
            st.session_state.pca_model = None
            st.session_state.analysis_options = ["KMeans", "PCA 2D"]
            st.session_state.k = 3
            # st.session_state.kmeans_colorscale = 'ice'
            st.session_state.pc_x = 'PC1'
            st.session_state.pc_y = 'PC2'
            st.session_state.last_update = time.time()
            st.session_state.update_counter = 0  # Contador para forçar atualizações

    def _display_header(self):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            # Tenta carregar a imagem, mas não falha se não encontrar
            img_base64 = image_to_base64('LQCTEAMhet.png')
            if img_base64:
                st.markdown(
                    f"""
                    <div style="text-align: center">
                        <a href="https://lqc.unb.br/" target="_blank">
                            <img src="data:image/png;base64,{img_base64}" 
                                 style="width: 200px; cursor: pointer; border-radius: 10px;">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.title("LQC Analytics")

    def load_data(self, path):
        try:
            st.session_state.original_data = pd.read_csv(path, index_col=0).dropna()
            if st.session_state.original_data.empty:
                st.error("Arquivo vazio após remoção de valores faltantes!")
                return False

            st.session_state.filtered_data = st.session_state.original_data.copy()
            st.session_state.excluded_vars = []
            st.session_state.removed_vars = []
            self._full_recalculation()
            st.session_state.update_counter += 1
            return True
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {str(e)}")
            return False

    def _full_recalculation(self):
        """Recalcula todos os modelos estatísticos"""
        try:
            # 1. Verifica dados filtrados
            if st.session_state.filtered_data.empty:
                st.error("Nenhuma variável disponível após exclusão!")
                return

            # 2. Recalcula PCA
            esc = StandardScaler()
            df_esc = esc.fit_transform(st.session_state.filtered_data)

            # 3. Atualiza modelo PCA
            n_components = min(df_esc.shape[0] - 1, df_esc.shape[1])
            if n_components < 1:
                st.error("Dados insuficientes para PCA")
                return

            st.session_state.pca_model = PCA(n_components=n_components)
            st.session_state.pca_model.fit(df_esc)

            # 4. Atualiza loadings
            pc_names = [f'PC{i}' for i in range(1, n_components + 1)]
            st.session_state.loadings_df = pd.DataFrame(
                st.session_state.pca_model.components_.T,
                index=st.session_state.filtered_data.columns,
                columns=pc_names
            )

            # 5. Atualiza opções de PC para os gráficos
            if pc_names:
                if st.session_state.pc_x not in pc_names:
                    st.session_state.pc_x = pc_names[0]
                if st.session_state.pc_y not in pc_names:
                    st.session_state.pc_y = pc_names[1] if len(pc_names) > 1 else pc_names[0]

            # 6. Atualiza timestamp
            st.session_state.last_update = time.time()

        except Exception as e:
            st.error(f"Erro no recálculo: {str(e)}")

    def apply_exclusions(self, vars_to_exclude):
        try:
            # Identifica todas as variáveis originais
            all_vars = list(st.session_state.original_data.columns)
            vars_to_keep = [v for v in all_vars if v not in vars_to_exclude]

            if not vars_to_keep:
                st.error("Mantenha pelo menos uma variável!")
                return False

            # Atualiza variáveis removidas
            newly_removed = set(vars_to_exclude) - set(st.session_state.excluded_vars)
            restored = set(st.session_state.excluded_vars) - set(vars_to_exclude)

            st.session_state.removed_vars = list(
                (set(st.session_state.removed_vars) | newly_removed) - restored
            )

            st.session_state.excluded_vars = vars_to_exclude
            st.session_state.filtered_data = st.session_state.original_data[vars_to_keep]

            self._full_recalculation()
            st.session_state.update_counter += 1
            return True
        except Exception as e:
            st.error(f"Erro ao excluir variáveis: {str(e)}")
            return False

    def restore_variables(self, vars_to_restore):
        try:
            if not vars_to_restore:
                return False

            st.session_state.excluded_vars = [v for v in st.session_state.excluded_vars if v not in vars_to_restore]
            st.session_state.removed_vars = [v for v in st.session_state.removed_vars if v not in vars_to_restore]

            success = self.apply_exclusions(st.session_state.excluded_vars)
            if success:
                st.success(f"Variáveis restauradas: {', '.join(vars_to_restore)}")
                time.sleep(0.5)
                st.rerun()
            return success
        except Exception as e:
            st.error(f"Erro ao restaurar variáveis: {str(e)}")
            return False

    def display_restore_controls(self):
        if not st.session_state.removed_vars:
            return

        with st.expander("📁 Variáveis Removidas (Clique para restaurar)", expanded=False):
            selected_restore = st.multiselect(
                "Selecione variáveis para restaurar",
                sorted(st.session_state.removed_vars),
                key=f"var_restore_select_{st.session_state.update_counter}"
            )

            if st.button("♻️ Restaurar Variáveis Selecionadas", key=f"restore_btn_{st.session_state.update_counter}"):
                self.restore_variables(selected_restore)

    def display_interface(self):
        # Sidebar
        with st.sidebar:
            st.header("Configurações")

            path = st.text_input(
                "Caminho do arquivo CSV:",
                key="csv_path",
                help="Insira o caminho completo para o arquivo CSV"
            )

            if path:
                if st.session_state.original_data is None:
                    if self.load_data(path):
                        st.success("Dados carregados com sucesso!")
                        st.rerun()
                    else:
                        st.stop()

                if st.session_state.filtered_data is not None:
                    max_k = len(st.session_state.filtered_data)
                    pc_options = [f'PC{i}' for i in range(1,
                                                          st.session_state.pca_model.n_components_ + 1)] if st.session_state.pca_model else []

                    new_k = st.slider(
                        "Número de Clusters (K)",
                        1, max(max_k, 1), st.session_state.k
                    )

                    if new_k != st.session_state.k:
                        st.session_state.k = new_k
                        st.session_state.update_counter += 1

                    if pc_options:
                        new_pc_x = st.selectbox(
                            "PC X",
                            pc_options,
                            index=pc_options.index(st.session_state.pc_x) if st.session_state.pc_x in pc_options else 0,
                            key=f"pc_x_{st.session_state.update_counter}"
                        )

                        new_pc_y = st.selectbox(
                            "PC Y",
                            pc_options,
                            index=pc_options.index(st.session_state.pc_y) if st.session_state.pc_y in pc_options else (
                                1 if len(pc_options) > 1 else 0),
                            key=f"pc_y_{st.session_state.update_counter}"
                        )

                        if new_pc_x != st.session_state.pc_x or new_pc_y != st.session_state.pc_y:
                            st.session_state.pc_x = new_pc_x
                            st.session_state.pc_y = new_pc_y
                            st.session_state.update_counter += 1

                    st.session_state.analysis_options = st.multiselect(
                        "Análises para Exibir",
                        ["KMeans", "PCA 2D", "PCA 2D Arrows", "Variância Explicada", "PCA 3D"],
                        default=st.session_state.analysis_options,
                        key=f"analysis_options_{st.session_state.update_counter}"
                    )

                    # Configurações para o Kmeans
                    # if "KMeans" in st.session_state.analysis_options:
                    #     st.write("---")
                    #     color_options = [
                    #         'ice', 'viridis', 'plasma', 'inferno', 'magma',
                    #         'blues', 'reds', 'greens', 'purples',
                    #         'bluered', 'rdbu', 'spectral', 'rainbow'
                    #     ]
                    #
                    #     new_colorscale = st.selectbox(
                    #         "🎨 Paleta de cores - Kmeans",
                    #         color_options,
                    #         index=color_options.index(
                    #             st.session_state.kmeans_colorscale) if st.session_state.kmeans_colorscale
                    #                                                    in color_options else 0,
                    #         key=f"kmeans_colors_{st.session_state.update_counter}",
                    #         help="Escolha a paleta de cores para o Kmeans"
                    #     )
                    #
                    #     if new_colorscale != st.session_state.kmeans_colorscale:
                    #         st.session_state_kmeans_colorscale = new_colorscale
                    #         st.session_state.update_counter += 1

                    if st.button("🔄 Resetar Análise"):
                        self.load_data(path)
                        st.rerun()

        # Corpo Principal
        if st.session_state.original_data is not None:
            self._display_loadings_table()
            self.display_restore_controls()
            self._display_analyses()

    def _display_loadings_table(self):
        if st.session_state.loadings_df is None:
            return

        with st.container():
            st.header("📊 Tabela de Loadings")

            # Cria cópia dos dados atuais com todas as variáveis originais
            all_vars = list(st.session_state.original_data.columns)
            current_loadings = st.session_state.loadings_df.copy()

            # Adiciona variáveis excluídas com valores zero
            for var in st.session_state.excluded_vars:
                if var not in current_loadings.index:
                    # Cria linha com zeros para variável excluída
                    zero_row = pd.Series(0.0, index=current_loadings.columns, name=var)
                    current_loadings.loc[var] = zero_row

            # Reordena para manter ordem original
            current_loadings = current_loadings.reindex([v for v in all_vars if v in current_loadings.index])

            # Adiciona coluna de exclusão
            current_loadings['Excluir'] = [v in st.session_state.excluded_vars for v in current_loadings.index]

            # Aplica o estilo de heatmap com background_gradient
            def apply_heatmap_style(df):
                # Seleciona apenas as colunas numéricas (PCs) para o gradient
                numeric_cols = df.select_dtypes(include=[np.number]).columns

                # Calcula valores absolutos apenas para as colunas numéricas
                abs_values_numeric = df[numeric_cols].abs()
                max_abs_value = abs_values_numeric.max().max()

                # Cria o styler base
                styled = df.style

                # Aplica o gradient apenas nas colunas numéricas usando valores absolutos
                for col in numeric_cols:
                    styled = styled.background_gradient(
                        cmap='Blues',
                        subset=[col],  # Uma coluna por vez
                        vmin=0,
                        vmax=max_abs_value,
                        gmap=abs_values_numeric[col]  # Valores absolutos da coluna específica
                    )

                # Formata os valores numéricos com 3 casas decimais
                styled = styled.format({
                    **{col: '{:.3f}' for col in numeric_cols}
                })

                # Destaca variáveis excluídas com cor de fundo diferente
                def highlight_excluded(row):
                    if row['Excluir']:
                        return ['background-color: #ffcccc; opacity: 0.7'] * len(row)
                    return [''] * len(row)

                styled = styled.apply(highlight_excluded, axis=1)

                return styled

            # Aplica o estilo e cria o data_editor
            styled_df = apply_heatmap_style(current_loadings)

            # Editor interativo com heatmap
            edited_df = st.data_editor(
                styled_df,
                disabled=current_loadings.columns[:-1].tolist(),  # Desabilita colunas de PCs
                use_container_width=True,
                column_config={
                    "Excluir": st.column_config.CheckboxColumn(
                        "🗑️ Excluir",
                        help="Marque para excluir variável da análise",
                        default=False,
                    ),
                    # Configuração para as colunas de PCs
                    **{col: st.column_config.NumberColumn(
                        col,
                        help=f"Loading da variável no {col}",
                        format="%.3f"
                    ) for col in current_loadings.columns[:-1]}
                },
                key=f"loadings_editor_{st.session_state.update_counter}",
                hide_index=False,
                height=min(600, (len(current_loadings) + 1) * 35 + 100)  # Altura dinâmica
            )

            # Detecta mudanças
            new_exclusions = edited_df[edited_df['Excluir']].index.tolist()
            current_exclusions = st.session_state.excluded_vars

            # Aplica mudanças se necessário
            if set(new_exclusions) != set(current_exclusions):
                if self.apply_exclusions(new_exclusions):
                    st.rerun()

            # Controles rápidos
            st.write("---")
            col_ctrl1, col_ctrl2 = st.columns(2)

            with col_ctrl1:
                if st.button("✅ Incluir Todas", key=f"include_all_{st.session_state.update_counter}"):
                    if self.apply_exclusions([]):
                        st.rerun()

            with col_ctrl2:
                if st.button("❌ Excluir Todas", key=f"exclude_all_{st.session_state.update_counter}"):
                    if self.apply_exclusions(all_vars[:-1]):  # Mantém pelo menos uma
                        st.rerun()

            # Informações úteis
            st.write("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Variáveis Ativas", len(st.session_state.filtered_data.columns))
            with col2:
                st.metric("❌ Variáveis Excluídas", len(st.session_state.excluded_vars))
            with col3:
                if st.session_state.pca_model:
                    total_var = st.session_state.pca_model.explained_variance_ratio_[:3].sum() * 100
                    st.metric("🎯 Variância PC1+PC2+PC3", f"{total_var:.1f}%")
            with col4:
                if st.session_state.loadings_df is not None:
                    numeric_cols = current_loadings.select_dtypes(include=[np.number]).columns
                    max_loading = abs(current_loadings[numeric_cols].values).max()
                    st.metric("🔥 Loading Máximo", f"{max_loading:.3f}")

            # Legenda do heatmap
            st.info(
                "💡 **Legenda:** Tons de azul mais intensos = loadings com **maior magnitude** (valor absoluto). Valores positivos e negativos com mesma magnitude terão a mesma cor, pois na PCA importa apenas o tamanho do vetor. Linhas destacadas em rosa = variáveis excluídas.")

    def _display_analyses(self):
        if st.session_state.pca_model is None or st.session_state.filtered_data is None:
            st.warning("⚠️ Dados não disponíveis para análise.")
            return

        # Layout responsivo para gráficos
        if "PCA 3D" in st.session_state.analysis_options:
            # Se temos PCA 3D, mostra ele em linha completa primeiro
            self._run_pca_3d()

            # Depois mostra os outros em colunas
            other_analyses = [a for a in st.session_state.analysis_options if a != "PCA 3D"]
            if other_analyses:
                cols = st.columns(2)
                for i, analysis in enumerate(other_analyses):
                    with cols[i % 2]:
                        if analysis == "KMeans":
                            self._run_kmeans()
                        elif analysis == "PCA 2D":
                            self._run_pca_2d()
                        elif analysis == "PCA 2D Arrows":
                            self._run_pca_2d_arrows()
                        elif analysis == "Variância Explicada":
                            self._run_variance_analysis()
        else:
            # Layout em colunas para todos os gráficos
            cols = st.columns(2)
            analysis_count = 0

            for analysis in st.session_state.analysis_options:
                with cols[analysis_count % 2]:
                    if analysis == "KMeans":
                        self._run_kmeans()
                    elif analysis == "PCA 2D":
                        self._run_pca_2d()
                    elif analysis == "PCA 2D Arrows":
                        self._run_pca_2d_arrows()
                    elif analysis == "Variância Explicada":
                        self._run_variance_analysis()
                analysis_count += 1

    def _run_kmeans(self):
        try:
            if st.session_state.filtered_data is None or st.session_state.filtered_data.empty:
                st.warning("⚠️ Nenhum dado disponível para KMeans")
                return

            # Pré-processamento
            filtered_data = st.session_state.filtered_data.copy()
            esc = StandardScaler()
            df_esc = esc.fit_transform(filtered_data)

            # Matriz de correlação
            df_esc_T = pd.DataFrame(df_esc.T, columns=filtered_data.index).corr()

            # KMeans
            centroids = Kmeans.centroids_begin(st.session_state.k, df_esc_T)
            Kmeans.centroid_attributes(df_esc_T, centroids)
            fig, _, _ = Kmeans.kmeans(
                df_esc_T[[df_esc_T.index[0], df_esc_T.index[-1]]],
                st.session_state.k
            )

            # Atualização do gráfico
            fig.update_layout(
                title=f"🎯 KMeans (K={st.session_state.k})<br><sub>Variáveis: {len(filtered_data.columns)}</sub>",
                height=400,
                title_x=0.5
            )
            fig.update_xaxes(showgrid=True)
            # fig.update_layout(coloraxis={'colorscale': st.session_state.kmeans_colorscale})

            st.plotly_chart(fig, use_container_width=True, key=f"kmeans_{st.session_state.update_counter}")

        except Exception as e:
            st.error(f"❌ Erro no KMeans: {str(e)}")

    def _run_pca_2d(self):
        try:
            if st.session_state.pca_model is None or st.session_state.filtered_data is None:
                st.warning("⚠️ Dados não disponíveis para PCA 2D")
                return

            # Transformação PCA
            filtered_data = st.session_state.filtered_data.copy()
            esc = StandardScaler()
            df_esc = esc.fit_transform(filtered_data)
            components = st.session_state.pca_model.transform(df_esc)

            # Preparação dos scores
            pc_columns = [f'PC{i}' for i in range(1, st.session_state.pca_model.n_components_ + 1)]
            scores = pd.DataFrame(
                components,
                index=filtered_data.index,
                columns=pc_columns
            )

            # Gráfico
            fig = px.scatter(
                scores,
                x=st.session_state.pc_x,
                y=st.session_state.pc_y,
                color=filtered_data.index,
                title=f"📈 PCA 2D ({st.session_state.pc_x} vs {st.session_state.pc_y})<br><sub>Variáveis: {len(filtered_data.columns)}</sub>",
                height=400
            )

            fig.update_layout(
                title_x=0.5,
                showlegend=False
            )
            fig.update_xaxes(showgrid=True)

            st.plotly_chart(fig, use_container_width=True, key=f"pca2d_{st.session_state.update_counter}")

        except Exception as e:
            st.error(f"❌ Erro no PCA 2D: {str(e)}")

    def _run_pca_3d(self):
        try:
            if st.session_state.filtered_data is None or st.session_state.filtered_data.empty:
                st.warning("⚠️ Dados insuficientes para PCA 3D")
                return

            filtered_data = st.session_state.filtered_data.copy()
            esc = StandardScaler()
            df_esc = esc.fit_transform(filtered_data)

            # PCA 3D específico
            pca_3d = PCA(n_components=min(3, df_esc.shape[1], df_esc.shape[0] - 1))
            components = pca_3d.fit_transform(df_esc)
            total_var = pca_3d.explained_variance_ratio_.sum() * 100

            # Prepara DataFrame para o gráfico
            n_components = components.shape[1]
            pc_names = [f'PC{i + 1}' for i in range(n_components)]
            scores_3d = pd.DataFrame(components, columns=pc_names)

            # Garante que temos pelo menos 3 componentes (duplicando se necessário)
            if n_components < 3:
                for i in range(n_components, 3):
                    scores_3d[f'PC{i + 1}'] = 0

            fig = px.scatter_3d(
                scores_3d,
                x='PC1', y='PC2', z='PC3',
                color=filtered_data.index,
                title=f"🎲 PCA 3D (Variância Total: {total_var:.1f}%)<br><sub>Variáveis: {len(filtered_data.columns)}</sub>",
                height=600
            )

            fig.update_layout(
                title_x=0.5,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True, key=f"pca3d_{st.session_state.update_counter}")

        except Exception as e:
            st.error(f"❌ Erro no PCA 3D: {str(e)}")

    def _run_variance_analysis(self):
        try:
            if st.session_state.pca_model is None:
                st.warning("⚠️ Modelo PCA não disponível")
                return

            explained = st.session_state.pca_model.explained_variance_ratio_ * 100
            cumulative = np.cumsum(explained)
            pc_names = [f'PC{i}' for i in range(1, len(explained) + 1)]

            # Gráfico de barras com linha cumulativa
            fig = px.bar(
                x=pc_names, y=explained,
                title=f"📊 Variância Explicada<br><sub>Variáveis: {len(st.session_state.filtered_data.columns)}</sub>",
                labels={'x': 'Componentes Principais', 'y': 'Variância Explicada (%)'},
                height=400
            )

            # Adiciona linha cumulativa
            fig.add_scatter(
                x=pc_names,
                y=explained,
                mode='lines+markers',
                name="Acumulada",
                yaxis="y2",
                line=dict(color='red', width=3)
            )

            fig.update_layout(
                title_x=0.5,
                yaxis2=dict(
                    title="Variância Acumulada (%)",
                    overlaying="y",
                    side="right"
                )
            )
            fig.update_xaxes(showgrid=True)

            st.plotly_chart(fig, use_container_width=True, key=f"variance_{st.session_state.update_counter}")
        except Exception as e:
            st.error(f"❌ Erro na análise de variância: {str(e)}")

    def _run_pca_2d_arrows(self):
        try:
            if st.session_state.pca_model is None or st.session_state.filtered_data is None:
                st.warning("⚠️ Dados não disponíveis para PCA 2D Arrows")
                return

            # Transformação PCA
            filtered_data = st.session_state.filtered_data.copy()
            esc = StandardScaler()
            df_esc = esc.fit_transform(filtered_data)
            components = st.session_state.pca_model.transform(df_esc)

            # Preparação dos scores
            pc_columns = [f'PC{i}' for i in range(1, st.session_state.pca_model.n_components_ + 1)]
            scores = pd.DataFrame(
                components,
                index=filtered_data.index,
                columns=pc_columns
            )

            # Obtenção dos loadings e loadings_matrix (como na versão original)
            loadings = pd.DataFrame(
                st.session_state.pca_model.components_.T,
                index=filtered_data.columns,
                columns=pc_columns
            )

            # Loadings matrix escalada pela variância explicada (como no código original)
            loadings_matrix = st.session_state.pca_model.components_.T * np.sqrt(
                st.session_state.pca_model.explained_variance_)

            # Configurações das setas (ajustáveis)
            arrowscale = 3.0  # Escala das setas
            arrowsize = 1.5  # Tamanho da cabeça da seta
            arrowhead = 2  # Estilo da cabeça

            # Obtem índices dos PCs selecionados
            try:
                pc_x_idx = pc_columns.index(st.session_state.pc_x)
                pc_y_idx = pc_columns.index(st.session_state.pc_y)
            except ValueError:
                st.error(f"PCs selecionados não disponíveis: {st.session_state.pc_x}, {st.session_state.pc_y}")
                return

            # Cria o gráfico base (scatter dos scores)
            fig = px.scatter(
                scores,
                x=st.session_state.pc_x,
                y=st.session_state.pc_y,
                text=filtered_data.index,
                color=filtered_data.index,
                title=f"🏹 PCA 2D com Vetores ({st.session_state.pc_x} vs {st.session_state.pc_y})<br><sub>Variáveis: {len(filtered_data.columns)}</sub>",
                height=500
            )

            # Personaliza o layout
            fig.update_layout(
                title_x=0.5,
                showlegend=False,
                yaxis=dict(tickfont=dict(size=14)),
                xaxis=dict(tickfont=dict(size=14)),
                font=dict(family="Arial", size=15, color='white')
            )

            fig.update_xaxes(title_font_family="Arial", title_font_size=16, showgrid=True)
            fig.update_yaxes(title_font_family="Arial", title_font_size=16)

            # Adiciona as setas para cada variável (loadings)
            for i, feature in enumerate(filtered_data.columns):
                # Coordenadas da seta
                arrow_x = loadings_matrix[i, pc_x_idx] * arrowscale
                arrow_y = loadings_matrix[i, pc_y_idx] * arrowscale

                # Adiciona a seta
                fig.add_annotation(
                    ax=0, ay=0,
                    axref="x", ayref="y",
                    x=arrow_x,
                    y=arrow_y,
                    showarrow=True,
                    arrowsize=arrowsize,
                    arrowhead=arrowhead,
                    arrowcolor="red",
                    arrowwidth=2,
                    xanchor="right",
                    yanchor="top"
                )

                # Adiciona o label da variável no final da seta
                fig.add_annotation(
                    x=arrow_x,
                    y=arrow_y,
                    ax=0, ay=0,
                    xanchor="center",
                    yanchor="bottom",
                    text=feature,
                    yshift=5,
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=12,
                        color="darkred"
                    ),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1
                )

            st.plotly_chart(fig, use_container_width=True, key=f"pca2d_arrows_{st.session_state.update_counter}")

            # Informações adicionais
            st.caption(
                f"🏹 Setas vermelhas representam as variáveis nos componentes {st.session_state.pc_x} e {st.session_state.pc_y}")
            st.caption(f"📏 Comprimento das setas = importância da variável | Direção = correlação com os PCs")

        except Exception as e:
            st.error(f"❌ Erro no PCA 2D Arrows: {str(e)}")


# ======================================
# Ponto de Entrada
# ======================================
def main():
    app = LQCAnalyticsApp()
    app.display_interface()


if __name__ == "__main__":
    main()

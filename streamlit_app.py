import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Hemithea Analiz", layout="wide")

# Mobil ve Görsel CSS
st.markdown("""
    <style>
    .main > div { padding: 1rem; }
    iframe { width: 100% !important; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. AYARLAR VE OTURUM ---
BASE_RENDER_URL = "https://apphemitheanetwork.onrender.com/uploads"

if "username" not in st.session_state:
    st.session_state.username = None
if "user_data" not in st.session_state:
    st.session_state.user_data = None

# URL Parametrelerini Yakala (Android'den otomatik geçiş)
query_params = st.query_params
if "username" in query_params and not st.session_state.username:
    st.session_state.username = query_params.get("username")
    st.session_state.token = query_params.get("token")

# --- 3. FONKSİYONLAR ---
@st.cache_data(ttl=2)
def load_dynamic_data(uname, token):
    if not uname or not token: return None
    try:
        target_url = f"{BASE_RENDER_URL}/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=10)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        return None
    except: return None

# --- 4. SIDEBAR (GİRİŞ & YÜKLEME) ---
with st.sidebar:
    st.title("🔍 Hemithea Panel")
    
    if not st.session_state.username:
        st.subheader("🔑 Giriş / Kayıt")
        u_id = st.text_input("Kullanıcı ID:")
        u_tk = st.text_input("Token / Şifre:", type="password")
        if st.button("Sistemi Başlat"):
            st.session_state.username = u_id
            st.session_state.token = u_tk
            st.rerun()
        st.info("💡 Kayıt için Android uygulamasını kullanabilir veya ID/Token belirleyerek girebilirsiniz.")
    else:
        st.success(f"Hoş geldin, {st.session_state.username}")
        
        # KENDİ VERİNİ ANALİZ ET
        st.divider()
        st.subheader("📤 Veri Yükle")
        up_file = st.file_uploader("CSV Dosyası Seç", type=["csv"])
        if up_file:
            st.session_state.user_data = pd.read_csv(up_file)
            st.toast("Veri yüklendi!", icon="✅")
            
        if st.button("🚪 Çıkış Yap"):
            st.session_state.username = None
            st.session_state.user_data = None
            st.rerun()

# --- 5. ANA AKIŞ ---
st.title("🌐 Hemithea Network Analytics")

# Veriyi belirle (Önce manuel yüklenen, sonra Render)
if st.session_state.user_data is not None:
    final_data = st.session_state.user_data
elif st.session_state.username:
    final_data = load_dynamic_data(st.session_state.username, st.session_state.token)
else:
    final_data = None

if final_data is not None:
    st.success("✅ Analiz Hazır!")
    
    # 1. ANALİZ HESAPLAMALARI
    G = nx.from_pandas_edgelist(final_data, source=final_data.columns[0], target=final_data.columns[1])
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    metrics_df = pd.DataFrame({
        'node': list(degree_cent.keys()),
        'degree': list(degree_cent.values()),
        'betweenness': list(betweenness.values())
    })

    # KNN Renklendirme
    if len(metrics_df) > 3:
        X = metrics_df[['degree', 'betweenness']].values
        y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
        X_scaled = StandardScaler().fit_transform(X)
        knn = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1)).fit(X_scaled, y)
        metrics_df['color'] = pd.Series(knn.predict(X_scaled)).map({1: "#e74c3c", 0: "#3498db"})

    # SEKME YAPISI
    tab1, tab2, tab3 = st.tabs(["🕸️ Ağ Haritası", "📈 Metrikler", "📄 Veri"])

    with tab1:
    if isinstance(data, pd.DataFrame):
        G = nx.from_pandas_edgelist(data, source=data.columns[0], target=data.columns[1])
        degree_cent = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)

        metrics_df = pd.DataFrame({
            'node': list(degree_cent.keys()),
            'degree': list(degree_cent.values()),
            'betweenness': list(betweenness.values())
        })

        metrics_df['color'] = np.where(
            metrics_df['betweenness'] > metrics_df['betweenness'].mean(),
            "#e74c3c", "#3498db"
        )

        st.subheader("🕸️ Etkileşim Haritası")
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
        for _, row in metrics_df.iterrows():
            net.add_node(row['node'], label=str(row['node']), color=row['color'])
        for edge in G.edges():
            net.add_edge(edge[0], edge[1])

        html_data = net.generate_html()
        components.html(html_data, height=550)

        


    with tab2:
        st.subheader("Ağ İstatistikleri (KNN)")
        st.dataframe(metrics_df, use_container_width=True)

    with tab3:
        st.subheader("Yüklenen Ham Veri")
        st.dataframe(final_data, use_container_width=True)

    # --- 6. İNDİRME BAĞLANTILARI ---
    st.divider()
    st.subheader("📥 Raporları İndir")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = metrics_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📄 CSV Raporu", csv, "hemithea_rapor.csv", "text/csv")
    with col2:
        st.download_button("🌐 HTML Haritası", html_data, "network.html", "text/html")
    with col3:
        fig, ax = plt.subplots(); nx.draw(G, with_labels=True, node_color="#3498db")
        buf = BytesIO(); plt.savefig(buf, format="png")
        st.download_button("🖼️ PNG Olarak Kaydet", buf.getvalue(), "network.png", "image/png")

else:
    st.info("👋 Hoş geldiniz! Lütfen giriş yapın veya kendi CSV dosyanızı yükleyin.")

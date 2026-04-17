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

# --- 1. OTURUM VE PARAMETRE YÖNETİMİ ---
if "username" not in st.session_state:
    st.session_state.username = None
if "user_data" not in st.session_state:
    st.session_state.user_data = None

params = st.query_params
if "username" in params and not st.session_state.username:
    st.session_state.username = params["username"]
    st.session_state.token = params.get("token", "")

# --- 2. GÜVENLİ VERİ ÇEKME (RENDER) ---
@st.cache_data(ttl=2)
def load_dynamic_data(uname, token):
    if not uname or not token: return None
    try:
        target_url = f"https://apphemitheanetwork.onrender.com/uploads/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=20)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df if not df.empty else "EMPTY"
        return "NOT_FOUND"
    except: return "CONNECTION_ERROR"

# --- 3. ANA EKRAN AYARLARI ---
st.set_page_config(page_title="Hemithea Analytics", layout="wide")
st.title("🌐 Hemithea Network Analytics")

# --- 4. SIDEBAR: GİRİŞ VE DOSYA YÜKLEME ---
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    if not st.session_state.username:
        st.subheader("🔑 Giriş")
        u_id = st.text_input("Kullanıcı ID:")
        u_tk = st.text_input("Token:", type="password")
        if st.button("Sistemi Başlat"):
            st.session_state.username = u_id
            st.session_state.token = u_tk
            st.rerun()
    else:
        st.success(f"Kullanıcı: {st.session_state.username}")
        
        # İŞTE BURASI: Kendi Verini Analiz Et Kısmı
        st.divider()
        st.subheader("📤 Kendi Verini Analiz Et")
        uploaded_file = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"])
        if uploaded_file:
            st.session_state.user_data = pd.read_csv(uploaded_file)
            st.toast("Kendi veriniz yüklendi!", icon="✅")

        if st.button("🚪 Oturumu Kapat"):
            st.session_state.username = None
            st.session_state.user_data = None
            st.rerun()

# --- 5. VERİ SEÇİM MANTIĞI ---
data_result = None

if st.session_state.username:
    # Öncelik: Kullanıcının o an yüklediği dosya
    if st.session_state.user_data is not None:
        data_result = st.session_state.user_data
    # İkinci Sırada: Render sunucusundaki kayıtlı dosya
    else:
        data_result = load_dynamic_data(st.session_state.username, st.session_state.token)
else:
    st.info("👋 Lütfen giriş yapın veya Android uygulamasından yönlenin.")
    st.stop()

# --- 6. ANALİZ VE GÖRSELLEŞTİRME ---
if isinstance(data_result, pd.DataFrame):
    G = nx.from_pandas_edgelist(data_result, source=data_result.columns[0], target=data_result.columns[1])
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    metrics_df = pd.DataFrame({
        'node': list(degree_cent.keys()),
        'degree': list(degree_cent.values()),
        'betweenness': list(betweenness.values())
    })

    # KNN ve Renklendirme
    if len(metrics_df) > 3:
        X = metrics_df[['degree', 'betweenness']].values
        y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
        X_scaled = StandardScaler().fit_transform(X)
        knn = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1)).fit(X_scaled, y)
        metrics_df['color'] = pd.Series(knn.predict(X_scaled)).map({1: "#e74c3c", 0: "#3498db"})

    # GRAFİK ÇİZİMİ
    st.subheader("🕸️ Etkileşim Haritası")
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    for _, row in metrics_df.iterrows():
        net.add_node(row['node'], label=str(row['node']), color=row.get('color', "#3498db"))
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    html_data = net.generate_html()
    components.html(html_data, height=550)

    # --- 7. İNDİRME BAĞLANTILARI ---
    st.divider()
    st.subheader("📥 Raporları İndir")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV İndir
        csv = metrics_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📄 CSV Raporu", csv, "hemithea_analiz.csv", "text/csv")
        
    with col2:
        # HTML Grafiği İndir
        st.download_button("🌐 HTML Haritası", html_data, "network.html", "text/html")
        
    with col3:
        # PNG (Matplotlib üzerinden statik kopya)
        fig, ax = plt.subplots()
        nx.draw(G, with_labels=True, node_color="#3498db", edge_color="#bdc3c7")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        st.download_button("🖼️ PNG Olarak Kaydet", buf.getvalue(), "network.png", "image/png")

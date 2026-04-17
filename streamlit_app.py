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

# 1. Sayfa Ayarları
st.set_page_config(page_title="Hemithea Analiz", layout="wide")

# Mobil CSS
st.markdown("""
    <style>
    .main > div { padding: 0.5rem; }
    iframe { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# --- AYARLAR ---
BASE_RENDER_URL = "https://apphemitheanetwork.onrender.com/uploads"

# --- 1. GİTHUB VİTRİN VE RAW DÖNÜŞTÜRÜCÜ ---
def to_raw(url):
    if "github.com" in url and "raw" not in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

@st.cache_data(ttl=600)
def get_vitrin_data(secim):
    linkler = {
        "Efendi Analizi": "https://github.com/seydanur-hemithea/appHemitheaNetwork2/blob/main/Efendi.csv",
        "Game of Thrones": "https://github.com/seydanur-hemithea/appHemitheaNetwork2/blob/main/GoT.csv"
    }
    try:
        raw_url = to_raw(linkler[secim])
        res = requests.get(raw_url)
        return pd.read_csv(StringIO(res.text))
    except: return pd.DataFrame({'Kaynak': ['Örnek'], 'Hedef': ['Veri']})

# 2. Dinamik Veri Yükleme Fonksiyonu
@st.cache_data(ttl=2)
def load_dynamic_data(uname, token): # Token parametresi eklendi
    if not uname or not token:
        return None
    try:
        # FastAPI'ye giderken token'ı query parametresi olarak ekliyoruz
        target_url = f"{BASE_RENDER_URL}/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=5)
        
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        elif response.status_code == 401:
            st.error("🔑 Yetkisiz Erişim: Anahtarınız geçersiz veya süresi dolmuş.")
            return None
        return None
    except Exception as e:
        return None

# --- ANA AKIŞ ---
st.title("🌐 Hemithea Network Analytics")

# URL parametresinden user_name al
query_params = st.query_params
current_username = query_params.get("username")
current_token = query_params.get("token")

# --- DEDEKTİF PANELİ (SIDEBAR) ---

if current_username:
    st.sidebar.title("🔍 Sistem Denetimi")
    st.sidebar.info(f"Kullanıcı Adı: {current_username}")
    
    if st.sidebar.button("🔗 Bağlantıyı Manuel Test Et"):
        try:
            # Test ederken de token kullanmalıyız
            test_url = f"{BASE_RENDER_URL}/{current_username}/network_data.csv?token={current_token}"
            res = requests.get(test_url, timeout=5)
            if res.status_code == 200:
                st.sidebar.success(f"📡 Güvenli bağlantı kuruldu!")
            else:
                st.sidebar.error(f"📡 Erişim Reddedildi! (Kod: {res.status_code})")
        except Exception as e:
            st.sidebar.error(f"Bağlantı Hatası: {e}")


# Veriyi yüklemeyi dene
data = load_dynamic_data(current_username, current_token)


if data is not None:
    # Sütunları belirle
    cols = data.columns.tolist()
    src = cols[0]
    tgt = cols[1]

    st.success(f"✅ Analiz Hazır!")
    
    # Sekmeler
    tab1, tab2, tab3 = st.tabs(["🕸️ Ağ Haritası", "📈 Metrikler", "📄 Veri"])

    G = nx.from_pandas_edgelist(data, source=src, target=tgt)

    with tab1:
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

    with tab2:
        st.subheader("Ağ İstatistikleri")
        degree_cent = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        metrics_df = pd.DataFrame({
            'Aktör': list(degree_cent.keys()),
            'Bağlantı Skoru': list(degree_cent.values()),
            'Stratejik Rol': list(betweenness.values())
        }).sort_values(by='Bağlantı Skoru', ascending=False)

        st.dataframe(metrics_df, use_container_width=True)

    with tab3:
        st.subheader("Yüklenen Veri Tablosu")
        st.dataframe(data, use_container_width=True)

else:
    st.info("👋 Hoş geldiniz! Analiz edilecek veri bekleniyor...")
    if not current_username or not current_token:
        st.warning("⚠️ Güvenlik uyarısı: Uygulama üzerinden güvenli giriş yapmanız gerekiyor.")
    else:
        st.warning(f"{current_username} için dosya bulunamadı veya erişim yetkiniz yok.")    
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

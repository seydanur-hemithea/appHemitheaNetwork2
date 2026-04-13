import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
from io import StringIO

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

# 2. Dinamik Veri Yükleme Fonksiyonu
@st.cache_data(ttl=2)
def load_dynamic_data(uid):
    if not uid:
        return None
    try:
        target_url = f"{BASE_RENDER_URL}/{uid}/network_data.csv"
        response = requests.get(target_url, timeout=5)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        return None
    except Exception as e:
        return None

# --- ANA AKIŞ ---
st.title("🌐 Hemithea Network Analytics")

# URL parametresinden user_id al
query_params = st.query_params
user_id = query_params.get("user_id")

# --- DEDEKTİF PANELİ (SIDEBAR) ---
if user_id:
    st.sidebar.title("🔍 Sistem Denetimi")
    test_url = f"{BASE_RENDER_URL}/{user_id}/network_data.csv"
    st.sidebar.info(f"Kullanıcı ID: {user_id}")
    
    if st.sidebar.button("🔗 Bağlantıyı Manuel Test Et"):
        try:
            res = requests.get(test_url, timeout=5)
            if res.status_code == 200:
                st.sidebar.success(f"📡 Dosya bulundu! (Kod: 200)")
            else:
                st.sidebar.error(f"📡 Dosya Yok! (Hata Kodu: {res.status_code})")
                st.sidebar.warning("İpucu: Android'in dosyayı doğru klasöre attığından emin ol.")
        except Exception as e:
            st.sidebar.error(f"Bağlantı Hatası: {e}")

# Veriyi yüklemeyi dene
data = load_dynamic_data(user_id)

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
        st.subheader("Etkileşimli Ağ Haritası")
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        net.toggle_physics(True)
        # Siyah ekranı önlemek için save_graph yöntemi daha güvenli olabilir ama şimdilik bunu deneyelim
        html_content = net.generate_html()
        components.html(html_content, height=650)

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
    if not user_id:
        st.warning("Uygulama üzerinden giriş yapmanız gerekiyor.")
    else:
        st.warning(f"ID {user_id} için sunucuda dosya bulunamadı. Lütfen Android'den dosya seçip gönderin.")

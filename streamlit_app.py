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

# 2. Dinamik Veri Yükleme (Android ID'sine Göre)
query_params = st.query_params
user_id = query_params.get("user_id")

# Render üzerindeki baz URL (Senin Render linkin)
BASE_RENDER_URL = "https://apphemitheanetwork.onrender.com/uploads"

@st.cache_data(ttl=2) # 2 saniyede bir tazele ki Android'den gelince hemen görünsün
def load_dynamic_data(uid):
    if user_id:
    test_url = f"{BASE_RENDER_URL}/{user_id}/network_data.csv"
    st.sidebar.info(f"🔍 Denenen URL: {test_url}")
    
    # Manuel bir kontrol butonu (opsiyonel ama hayat kurtarır)
    if st.sidebar.button("Bağlantıyı Manuel Test Et"):
        res = requests.get(test_url)
        st.sidebar.write(f"📡 Sunucu Durum Kodu: {res.status_code}")
        if res.status_code == 200:
            st.sidebar.success("Dosya bulundu!")
        else:
            st.sidebar.error("Dosya bulunamadı (404) veya erişim yasak (403).")


    
    if not uid:
        return None
    try:
        # Android'in gönderdiği tam yol: uploads/{user_id}/network_data.csv
        target_url = f"{BASE_RENDER_URL}/{uid}/network_data.csv"
        response = requests.get(target_url)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        return None
    except:
        return None

st.title("🌐 Hemithea Network Analytics")

# Veriyi yükle
data = load_dynamic_data(user_id)

if data is not None:
    # 3. Sütun İsimlerini Esnek Yapalım (Hata almamak için)
    # Eğer CSV'de 'Source' yoksa ilk iki sütunu kullan
    cols = data.columns.tolist()
    src = cols[0]
    tgt = cols[1]

    st.success(f"✅ Analiz Hazır! (Kullanıcı: {user_id})")
    
    # 4. Sekmeli Görünüm (Zekice Tasarım)
    tab1, tab2, tab3 = st.tabs(["🕸️ Ağ Haritası", "📈 Metrikler", "📄 Veri"])

    G = nx.from_pandas_edgelist(data, source=src, target=tgt)

    with tab1:
        # Pyvis Görselleştirme
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        
        # Fizik motoru ayarları (Daha ferah görünüm)
        net.toggle_physics(True)
        html_content = net.generate_html()
        components.html(html_content, height=650)

    with tab2:
        # Analitik Metrikler
        degree_cent = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        metrics_df = pd.DataFrame({
            'Aktör': list(degree_cent.keys()),
            'Etki Gücü': list(degree_cent.values()),
            'Köprü Rolü': list(betweenness.values())
        }).sort_values(by='Etki Gücü', ascending=False)

        col1, col2 = st.columns(2)
        col1.metric("Toplam Düğüm", len(G.nodes))
        col2.metric("En Popüler", metrics_df.iloc[0]['Aktör'])
        
        st.dataframe(metrics_df, use_container_width=True)

    with tab3:
        st.dataframe(data, use_container_width=True)

else:
    # Veri yoksa kullanıcıyı karşılayan ekran
    st.info("👋 Hoş geldiniz! Analiz edilecek veri bekleniyor...")
    st.warning("Lütfen Android uygulamasından bir dosya yükleyin.")
    # Debug için ID'yi göster
    if user_id:
        st.write(f"Aranan ID: {user_id}")
        st.write(f"Beklenen Yol: {BASE_RENDER_URL}/{user_id}/network_data.csv")

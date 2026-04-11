import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Sayfa Ayarları (Mobil için geniş ekran ve başlık)
st.set_page_config(page_title="Hemithea Analiz", layout="wide")

# Mobil için CSS Dokunuşu (Kenar boşluklarını sıfırlıyoruz)
st.markdown("""
    <style>
    .main > div { padding: 0rem 0.5rem; }
    iframe { width: 100% !important; height: 500px !important; }
    </style>
    """, unsafe_allow_escaping=True)

st.title("📊 Ağ Analiz Paneli")

# RENDER ÜZERİNDEKİ DOSYAYI OKUMA
# Not: Render linkini buraya tam olarak yazmalısın
RENDER_CSV_URL = "https://hemithea-backend.onrender.com/uploads/data.csv"

@st.cache_data(ttl=5) # Her 5 saniyede bir yeni veri var mı diye bakabilir
def load_data():
    try:
        # Doğrudan Render linkinden oku
        df = pd.read_csv(RENDER_CSV_URL)
        return df
    except Exception as e:
        return None

data = load_data()

if data is not None:
    st.success("Yeni Veri Algılandı!")
    
    # Ağ Analizi Başlasın
    G = nx.from_pandas_edgelist(data, source='Source', target='Target')
    
    # Görselleştirme (Pyvis ile Mobil Uyumlu)
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    
    # Mobil için etkileşim ayarları
    net.set_options("""
    var options = {
      "physics": { "enabled": true, "stabilization": { "enabled": true } },
      "interaction": { "zoomView": true, "dragView": true }
    }
    """)

    # HTML olarak render et
    html_data = net.generate_html()
    components.html(html_data, height=550)
    
    # Küçük bir özet tablo
    st.subheader("Veri Özeti")
    st.dataframe(data.head(5), use_container_width=True)

else:
    st.warning("Henüz analiz edilecek bir veri yüklenmedi. Lütfen uygulamadan dosya seçin.")

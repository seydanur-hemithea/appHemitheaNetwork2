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
    """, unsafe_allow_html=True)

st.title("📊 Ağ Analiz Paneli")

# RENDER ÜZERİNDEKİ DOSYAYI OKUMA
# Not: Render linkini buraya tam olarak yazmalısın
RENDER_CSV_URL = "https://apphemitheanetwork.onrender.com/uploads/data.csv"

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

#end
if data is not None:
    G = nx.from_pandas_edgelist(data, source='Source', target='Target')
    
    # 1. HESAPLAMALAR (Metrikler)
    # Derece Merkeziliği: Kimin kaç bağlantısı var?
    degree_cent = nx.degree_centrality(G)
    # Arasındalık Merkeziliği: Kim köprü görevi görüyor?
    betweenness = nx.betweenness_centrality(G)
    
    # Metrikleri DataFrame yapalım
    metrics_df = pd.DataFrame({
        'Düğüm': list(degree_cent.keys()),
        'Bağlantı Gücü': list(degree_cent.values()),
        'Köprü Rolü': list(betweenness.values())
    }).sort_values(by='Bağlantı Gücü', ascending=False)

    # 2. SEKMELİ GÖRÜNÜM (Mobil için çok daha temizdir)
    tab1, tab2, tab3 = st.tabs(["🌐 Ağ Grafiği", "📊 İstatistikler", "📋 Veri"])

    with tab1:
        st.subheader("Etkileşimli Ağ Haritası")
        net = Network(height="500px", width="100%", bgcolor="#ffffff")
        net.from_nx(G)
        html_data = net.generate_html()
        components.html(html_data, height=550)

    with tab2:
        st.subheader("Ağ Metrikleri")
        # En önemli düğümleri gösteren bir özet kartı
        top_node = metrics_df.iloc[0]['Düğüm']
        st.metric(label="En Merkezi Aktör", value=top_node)
        
        st.write("Düğüm Bazlı Analiz:")
        st.dataframe(metrics_df, use_container_width=True)

    with tab3:
        st.subheader("Ham Veri Seti")
        st.dataframe(data, use_container_width=True)

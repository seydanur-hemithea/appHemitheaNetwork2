import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
from io import StringIO, BytesIO
import matplotlib
# KRİTİK: Render üzerinde hata almamak için Agg backend kullanıyoruz
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

# 1. Sayfa Ayarları
st.set_page_config(page_title="Hemithea Analiz", layout="wide")

# Mobil CSS
st.markdown("""
    <style>
    .main > div { padding: 0.5rem; }
    iframe { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

BASE_RENDER_URL = "https://apphemitheanetwork.onrender.com/uploads"

@st.cache_data(ttl=2)
def load_dynamic_data(uname, token):
    if not uname or not token:
        return None
    try:
        target_url = f"{BASE_RENDER_URL}/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=5)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        return None
    except:
        return None

# --- ANA AKIŞ ---
st.title("🌐 Hemithea Network Analytics")

query_params = st.query_params
current_username = query_params.get("username")
current_token = query_params.get("token")

data = load_dynamic_data(current_username, current_token)

if data is not None:
    cols = data.columns.tolist()
    src, tgt = cols[0], cols[1]
    st.success(f"✅ Analiz Hazır!")
    
    tab1, tab2, tab3 = st.tabs(["🕸️ Ağ Haritası", "📈 Metrikler", "📄 Veri"])
    G = nx.from_pandas_edgelist(data, source=src, target=tgt)

    with tab1:
        st.subheader("Etkileşimli Ağ Haritası")
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        net.toggle_physics(True)
        components.html(net.generate_html(), height=650)

        st.divider()
        # --- GÜVENLİ FOTOĞRAF OLUŞTURMA ---
        if st.button("📸 İndirme Butonunu Hazırla"):
            try:
                plt.clf()
                fig, ax = plt.subplots(figsize=(10, 7))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', width=1.0)
                
                buf = BytesIO()
                plt.savefig(buf, format="png")
                st.download_button(label="📥 PNG Olarak İndir", data=buf.getvalue(), file_name="graph.png", mime="image/png")
                plt.close(fig)
            except Exception as e:
                st.error(f"Grafik hatası: {e}")

    with tab2:
        st.subheader("Ağ İstatistikleri")
        degree_cent = nx.degree_centrality(G)
        metrics_df = pd.DataFrame({
            'Aktör': list(degree_cent.keys()),
            'Skor': list(degree_cent.values())
        }).sort_values(by='Skor', ascending=False)
        
        st.dataframe(metrics_df, use_container_width=True)
       csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig') # 'utf-8-sig' Excel uyumluluğu sağlar

st.download_button(
    label="📄 Metrikleri Tablo (CSV) Olarak İndir",
    data=csv_data,
    file_name=f"hemithea_metrics.csv",
    mime="text/csv"
)

    with tab3:
        st.dataframe(data, use_container_width=True)
else:
    st.info("👋 Veri bekleniyor...")

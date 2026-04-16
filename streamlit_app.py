import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
from io import StringIO, BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
# YZ kütüphanelerini ekleyelim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Sayfa Ayarları
st.set_page_config(page_title="Hemithea Analiz", layout="wide")

BASE_RENDER_URL = "https://apphemitheanetwork.onrender.com/uploads"

@st.cache_data(ttl=2)
def load_dynamic_data(uname, token):
    if not uname or not token: return None
    try:
        target_url = f"{BASE_RENDER_URL}/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=5)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        return None
    except: return None

# --- ANA AKIŞ ---
st.title("🌐 Hemithea Network Analytics")

current_username = st.query_params.get("username")
current_token = st.query_params.get("token")

data = load_dynamic_data(current_username, current_token)

if data is not None:
    src, tgt = data.columns[0], data.columns[1]
    st.success(f"✅ Veri Bağlantısı Kuruldu")
    
    tab1, tab2, tab3 = st.tabs(["🕸️ Ağ Haritası", "🤖 Yapay Zeka Analizi", "📄 Veri"])
    G = nx.from_pandas_edgelist(data, source=src, target=tgt)

    with tab1:
        st.subheader("Etkileşimli Ağ Haritası")
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        components.html(net.generate_html(), height=550)
        
        st.divider()
        # KARARMA ÖNLEYİCİ: Çizimi butona bağladık
        if st.button("📸 Statik Grafik Oluştur"):
            with st.spinner("YZ görselleştirme hazırlanıyor..."):
                plt.clf()
                fig, ax = plt.subplots(figsize=(8, 6))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=500, width=0.7, font_size=7)
                buf = BytesIO()
                plt.savefig(buf, format="png", dpi=100)
                buf.seek(0)
                st.download_button("📥 PNG İndir", buf, "graph.png", "image/png")
                plt.close(fig)

    with tab2:
        st.subheader("🤖 KNN Gruplandırma ve Metrik Analizi")
        
        # Temel Metrikler
        degree_cent = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        metrics_df = pd.DataFrame({
            'Aktör': list(degree_cent.keys()),
            'Baglanti_Skoru': list(degree_cent.values()),
            'Stratejik_Rol': list(betweenness.values())
        })

        # KNN ANALİZİ - Sadece kullanıcı istediğinde çalışır, sistemi yormaz
        if st.checkbox("Yapay Zekayı Çalıştır (KNN Sınıflandırma)"):
            # Etiketleme: Ortalamanın üstündekiler 'Kritik' (1), altındakiler 'Normal' (0)
            avg_score = metrics_df['Baglanti_Skoru'].mean()
            y = (metrics_df['Baglanti_Skoru'] > avg_score).astype(int)
            X = metrics_df[['Baglanti_Skoru', 'Stratejik_Rol']].values
            
            if len(X) > 3:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                knn = KNeighborsClassifier(n_neighbors=min(3, len(X)-1))
                knn.fit(X_scaled, y)
                metrics_df['AI_Kategori'] = knn.predict(X_scaled)
                metrics_df['AI_Yorum'] = metrics_df['AI_Kategori'].map({1: "Kritik Düğüm", 0: "Uç Nokta"})
                
                st.info("KNN Modeli eğitildi ve aktörler sınıflandırıldı.")
            else:
                st.warning("YZ için veri yetersiz.")

        st.dataframe(metrics_df, use_container_width=True)
        
        csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📄 Sonuçları İndir (CSV)", csv_data, "hemithea_ai_results.csv", "text/csv")

    with tab3:
        st.dataframe(data, use_container_width=True)
else:
    st.info("👋 Veri bekleniyor...")

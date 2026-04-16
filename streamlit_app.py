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

@st.cache_data(ttl=2) # ttl=2 veri güncelliği için iyi ama test için artırılabilir
def load_dynamic_data(uname, token):
    if not uname or not token:
        st.error("Kullanıcı adı veya Token eksik!")
        return None
    try:
        target_url = f"{BASE_RENDER_URL}/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=10) # Süreyi 10'a çıkardık
        
        if response.status_code == 200:
            if not response.text.strip(): # Dosya boş mu kontrolü
                st.warning("Dosya bulundu ama içi boş!")
                return None
            return pd.read_csv(StringIO(response.text))
        else:
            st.error(f"Sunucu Hatası: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error("Sunucu çok geç yanıt verdi (Timeout)!")
        return None
    except Exception as e:
        st.error(f"Bağlantı Hatası: {e}")
        return None
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
        st.subheader("🤖 YZ Sınıflandırmalı Ağ Haritası (KNN)")
        
        # 1. Feature Engineering (Metrikler)
        degree_cent = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        metrics_df = pd.DataFrame({
            'node': list(degree_cent.keys()),
            'degree': list(degree_cent.values()),
            'betweenness': list(betweenness.values())
        })

        # 2. KNN ve Sınıflandırma
        # Sadece yeterli veri varsa ve kullanıcı istiyorsa çalıştır
        X = metrics_df[['degree', 'betweenness']].values
        if len(X) > 3:
            # KNN Eğit (Basit Sınıflandırma: Betweenness Ortalamasının üstü 'Stratejik')
            avg_bet = metrics_df['betweenness'].mean()
            y = (metrics_df['betweenness'] > avg_bet).astype(int)
            knn = KNeighborsClassifier(n_neighbors=3)
            # Basit eğitim
            knn.fit(X, y)
            metrics_df['AI_Role'] = knn.predict(X)
            
            # 3. Görselleştirme (Renk Atama)
            # KNN Sonucuna göre renkleri belirle
            color_map = {1: "red", 0: "skyblue"} # 1: Stratejik, 0: Normal
            
            # Pyvis Network oluştur
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            
            # Düğümleri tek tek özellikleriyle ekle
            for _, row in metrics_df.iterrows():
                node_label = row['node']
                node_color = color_map[row['AI_Role']]
                node_title = f"Rol: {'Stratejik' if row['AI_Role'] == 1 else 'Normal'}"
                
                # Pyvis'e düğümü renk ve başlık (hover) ile ekle
                net.add_node(node_label, label=node_label, color=node_color, title=node_title, size=20)
            
            # Kenarları ekle
            net.from_nx(G)
            net.toggle_physics(True)
            components.html(net.generate_html(), height=650)
            st.info("🎨 Düğümler KNN sonuçlarına göre renklendirildi (Kırmızı: Stratejik Köprü, Mavi: Normal).")
            
        else:
            st.warning("YZ Analizi için veri yetersiz. Standart grafik gösteriliyor.")
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            net.from_nx(G)
            components.html(net.generate_html(), height=650)
       

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

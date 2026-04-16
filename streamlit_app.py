import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import requests
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import tempfile
import os

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Hemithea Network AI", layout="wide")

# --- 2. VERİ ÇEKME FONKSİYONU ---
def load_dynamic_data(user_id):
    # Senin mevcut veri çekme mantığın buraya gelecek
    # Örnek olarak URL veya Google Sheets bağlantını buraya uyarla
    try:
        # Burayı kendi veri kaynağına göre güncelle Şeyda Nur
        # data_result = requests.get(URL).json() vb.
        return "EMPTY" # Şimdilik placeholder
    except:
        return "CONNECTION_ERROR"

# --- 3. ANA AKIŞ ---
current_user = "SeydaNur" # Dinamik kullanıcı adın
data_result = load_dynamic_data(current_user)

# NOT: data_result'ın DataFrame olduğunu varsayarak devam ediyorum 
# (Test için boş bir DF oluşturuyorum, sen kendi yükleme kısmını koru)
if data_result == "EMPTY":
    # Test verisi (Senin gerçek verilerin gelince burası DF olacak)
    data_result = pd.DataFrame({'Kaynak': ['A', 'B', 'C', 'A'], 'Hedef': ['B', 'C', 'A', 'D']})

if isinstance(data_result, pd.DataFrame):
    st.success(f"✅ Hemithea Sistemi Aktif: {current_user}")
    
    # --- ANALİZ MOTORU ---
    G = nx.from_pandas_edgelist(data_result, source=data_result.columns[0], target=data_result.columns[1])
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    metrics_df = pd.DataFrame({
        'node': list(degree_cent.keys()),
        'degree': list(degree_cent.values()),
        'betweenness': list(betweenness.values())
    })

    # --- KNN YAPAY ZEKA ---
    if len(metrics_df) > 3:
        try:
            X = metrics_df[['degree', 'betweenness']].values
            y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            knn = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1))
            knn.fit(X_scaled, y)
            metrics_df['AI_Role'] = knn.predict(X_scaled)
            metrics_df['Durum'] = metrics_df['AI_Role'].map({1: "🎯 Stratejik Köprü", 0: "👤 Normal Aktör"})
            metrics_df['color'] = metrics_df['AI_Role'].map({1: "#e74c3c", 0: "#3498db"})
        except:
            metrics_df['Durum'] = "Analiz Edildi"
            metrics_df['color'] = "#3498db"

    # --- 🌐 GÖRSELLEŞTİRME (İnteraktif Ağ) ---
    st.subheader("🌐 Ağ Etkileşim Haritası")
    try:
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
        for _, row in metrics_df.iterrows():
            net.add_node(row['node'], label=str(row['node']), color=row.get('color', "#3498db"))
        for edge in G.edges():
            net.add_edge(edge[0], edge[1])

        net.toggle_physics(True)
        html_data = net.generate_html()
        
        # MODERN YÖNTEM: st.iframe (Kararmayı önler ve Haziran 2026'ya hazır)
        st.iframe(srcdoc=html_data, height=550, scrolling=False)
    except Exception as e:
        st.error(f"Grafik yüklenemedi: {e}")

    # --- 📊 RAPORLAMA ---
    st.divider()
    st.subheader("📊 Analitik Raporlama")
    st.dataframe(metrics_df[['node', 'degree', 'betweenness', 'Durum']], width="stretch")

    # --- 📂 İNDİRME BÖLÜMÜ (Hatasız c1-c4 Yapısı) ---
    st.write("📂 **Analiz Çıktılarını İndir**")
    
    # 1. Satır: CSV ve Bilgi
    col1, col2 = st.columns(2)
    with col1:
        csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📄 Veri Raporu (CSV)", csv_data, f"analiz

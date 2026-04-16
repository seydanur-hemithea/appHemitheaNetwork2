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
import tempfile
import os

# --- 1. OTURUM VE PARAMETRE YÖNETİMİ ---
if "username" not in st.session_state:
    st.session_state.username = None

params = st.query_params
if "username" in params:
    st.session_state.username = params["username"]
    st.session_state.token = params.get("token", "")

# --- 2. GÜVENLİ VERİ ÇEKME ---
@st.cache_data(ttl=2)
def load_dynamic_data(uname, token):
    if not uname or not token:
        return None
    try:
        target_url = f"https://apphemitheanetwork.onrender.com/uploads/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=20)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df if not df.empty else "EMPTY"
        return "NOT_FOUND"
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return "CONNECTION_ERROR"
    except:
        return "ERROR"

# --- 3. ANA AKIŞ ---
st.title("🌐 Hemithea Network Analytics")

current_user = st.session_state.get("username")
current_token = st.session_state.get("token")

if not current_user:
    st.warning("🔑 Giriş bilgileri bekleniyor... Android üzerinden yönlendirme gerekiyor.")
    if st.button("🔄 Oturumu Yenile"):
        st.rerun()
    st.stop()

data_result = load_dynamic_data(current_user, current_token)

# --- 4. GÖRSELLEŞTİRME VE ANALİZ AKIŞI ---

if isinstance(data_result, pd.DataFrame):
    st.success(f"✅ Analiz Hazır: {current_user}")
    
    # 1. ANALİZLER (G ve Metrikler)
    G = nx.from_pandas_edgelist(data_result, source=data_result.columns[0], target=data_result.columns[1])
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    metrics_df = pd.DataFrame({
        'node': list(degree_cent.keys()),
        'degree': list(degree_cent.values()),
        'betweenness': list(betweenness.values())
    })

    # 2. KNN ANALİZİ (Grafik Renkleri İçin)
    if len(metrics_df) > 3:
        try:
            X = metrics_df[['degree', 'betweenness']].values
            y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            knn = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1))
            knn.fit(X_scaled, y)
            metrics_df['AI_Kategori'] = knn.predict(X_scaled)
            metrics_df['color'] = metrics_df['AI_Kategori'].map({1: "#e74c3c", 0: "#3498db"})
            metrics_df['Durum'] = metrics_df['AI_Kategori'].map({1: "🎯 Stratejik", 0: "👤 Normal"})
        except:
            metrics_df['color'] = "#3498db"
            metrics_df['Durum'] = "Analiz Edildi"

    # 3. GRAFİK ÇİZİMİ (İşte Burası Önemli!)
    st.subheader("🌐 İnteraktif Ağ Haritası")
    
    try:
        # Notebook=False ayarı Android WebView için daha stabildir
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Düğümleri KNN renkleriyle ekle
        for _, row in metrics_df.iterrows():
            node_color = row['color'] if 'color' in row else "#3498db"
            net.add_node(row['node'], label=str(row['node']), color=node_color)
        
        # Kenarları ekle
        for edge in G.edges():
            net.add_edge(edge[0], edge[1])

        net.toggle_physics(True) # Hareketli olmasını sağlar
        
        # KRİTİK: generate_html() içeriğini Android'in anlayacağı en saf haliyle gönderiyoruz
        html_data = net.generate_html()
        
        # Eski components.html yerine daha güvenli kapsayıcı
        st.components.v1.html(html_data, height=550, scrolling=False)
        
    except Exception as e:
        st.error(f"Grafik çizilirken hata: {e}")

    # 4. TABLO VE İNDİRME
    st.divider()
    st.dataframe(metrics_df[['node', 'degree', 'betweenness', 'Durum']], width="stretch")
    
    # İndirme butonları
    c1, c2 = st.columns(2)
    with c1:
        csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📄 CSV Raporu", csv_data, f"rapor_{current_user}.csv")
    with c2:
        st.info("💡 Grafiği yukarıdan inceleyebilirsiniz.")

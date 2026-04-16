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

# --- 4. DURUM KONTROLLERİ VE ANALİZ AKIŞI ---

if isinstance(data_result, pd.DataFrame):
    st.success(f"✅ Veri Bağlantısı Tamam: {current_user}")
    
    # 1. ANALİZLER (G ve Metrikler)
    G = nx.from_pandas_edgelist(data_result, source=data_result.columns[0], target=data_result.columns[1])
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    metrics_df = pd.DataFrame({
        'node': list(degree_cent.keys()),
        'degree': list(degree_cent.values()),
        'betweenness': list(betweenness.values())
    })

    # 2. KNN MOTORU
    if len(metrics_df) > 3:
        try:
            X = metrics_df[['degree', 'betweenness']].values
            y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            knn = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1))
            knn.fit(X_scaled, y)
            metrics_df['Rol_Tanimi'] = knn.predict(X_scaled)
            metrics_df['Durum'] = metrics_df['Rol_Tanimi'].map({1: "🎯 Stratejik", 0: "👤 Normal"})
        except: pass

    # 3. GÖRSELLEŞTİRME (Alt Alta, En Hafif Mod)
    st.subheader("📊 Analitik Sonuçlar")
    st.dataframe(metrics_df[['node', 'degree', 'betweenness', 'Durum']], width="stretch")
    
    st.divider()
    
    # --- İNDİRME BÖLÜMÜ (Kolon Hataları Düzeltildi) ---
    st.write("📂 **Analiz Çıktılarını İndir**")
    
    # İlk Satır Kolonlar: CSV ve HTML
    c1, c2 = st.columns(2)
    with c1:
        csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📄 Veri Raporu (CSV)", csv_data, f"analiz_{current_user}.csv")
    with c2:
        try:
            net_dl = Network(height="600px", width="100%")
            net_dl.from_nx(G)
            temp_path = os.path.join(tempfile.gettempdir(), f"map_{current_user}.html")
            net_dl.save_graph(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                html_string = f.read()
            st.download_button("🌐 İnteraktif Harita (HTML)", html_string, "harita.html", "text/html")
        except: st.write("Dosya Hazırlanıyor...")

    # İkinci Satır Kolonlar: Resimler (C3 Sorunu Buradaydı!)
    st.write("📸 **Görsel Kayıtlar (PNG)**")
    c3, c4 = st.columns(2) # C3 ve C4 burada tanımlandı!
    
    with c3:
        try:
            plt.clf()
            fig_tbl, ax_tbl = plt.subplots(figsize=(8, 5))
            ax_tbl.axis('off')
            ax_tbl.table(cellText=metrics_df.head(10).values, colLabels=metrics_df.columns, loc='center')
            buf_tbl = BytesIO()
            plt.savefig(buf_tbl, format="png")
            st.download_button("🖼️ Tablo Resmi", buf_tbl.getvalue(), "tablo.png")
            plt.close(fig_tbl)
        except: st.write("PNG Hazırlanıyor...")
        
    with c4:
        try:
            plt.clf()
            fig_gr, ax_gr = plt.subplots(figsize=(8, 5))
            nx.draw(G, with_labels=True, node_color='#3498db', node_size=300, font_size=6)
            buf_gr = BytesIO()
            plt.savefig(buf_gr, format="png")
            st.download_button("📸 Ağ Resmi", buf_gr.getvalue(), "ag.png")
            plt.close(fig_gr)
        except: st.write("PNG Hazırlanıyor...")

elif isinstance(data_result, str):
    st.warning(f"Bağlantı Durumu: {data_result}")
    if st.button("🔄 Tekrar Dene"): st.rerun()

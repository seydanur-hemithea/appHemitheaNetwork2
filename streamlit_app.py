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

# --- 1. GİTHUB VİTRİN VE RAW DÖNÜŞTÜRÜCÜ ---
def to_raw(url):
    if "github.com" in url and "raw" not in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

@st.cache_data(ttl=600)
def get_vitrin_data(secim):
    linkler = {
        "Efendi Analizi": "https://github.com/seydanur-hemithea/appHemitheaNetwork2/blob/main/Efendi.csv",
        "Game of Thrones": "https://github.com/seydanur-hemithea/appHemitheaNetwork2/blob/main/GoT.csv"
    }
    try:
        raw_url = to_raw(linkler[secim])
        res = requests.get(raw_url)
        return pd.read_csv(StringIO(res.text))
    except: return pd.DataFrame({'Kaynak': ['Örnek'], 'Hedef': ['Veri']})

# --- 2. OTURUM VE PARAMETRE YÖNETİMİ ---
if "username" not in st.session_state:
    st.session_state.username = None

# Android'den gelen linki yakala
params = st.query_params
if "username" in params and not st.session_state.username:
    st.session_state.username = params["username"]
    st.session_state.token = params.get("token", "")

# --- 3. GÜVENLİ VERİ ÇEKME (RENDER) ---
@st.cache_data(ttl=2)
def load_dynamic_data(uname, token):
    if not uname or not token: return None
    try:
        target_url = f"https://apphemitheanetwork.onrender.com/uploads/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=20)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df if not df.empty else "EMPTY"
        return "NOT_FOUND"
    except: return "CONNECTION_ERROR"

# --- 4. ANA AKIŞ VE GİRİŞ EKRANI ---
st.set_page_config(page_title="Hemithea Network", layout="wide")
st.title("🌐 Hemithea Network Analytics")

if not st.session_state.username:
    # --- VİTRİN VE GİRİŞ MODU ---
    st.info("👋 **Hemithea'ya Hoş Geldiniz!** Kendi analizleriniz için giriş yapın veya örnekleri inceleyin.")
    
    with st.sidebar:
        st.subheader("🛡️ Üye Paneli")
        mode = st.radio("İşlem Seçin:", ["Giriş Yap", "Kayıt Ol (Android)"])
        
        u_id = st.text_input("Kullanıcı ID:")
        u_tk = st.text_input("Token:", type="password")
        
        if st.button("Sistemi Başlat"):
            if u_id and u_tk:
                st.session_state.username = u_id
                st.session_state.token = u_tk
                st.rerun()
            else: st.warning("Lütfen bilgileri girin.")
        
        if mode == "Kayıt Ol (Android)":
            st.caption("✨ Kayıt işlemleri yakında doğrudan buradan da yapılabilecek. Şimdilik Android kimliğinizi kullanın.")

    # Vitrin Seçimi
    v_secim = st.radio("Örnek Analizleri İnceleyin:", ["Efendi Analizi", "Game of Thrones"], horizontal=True)
    data_result = get_vitrin_data(v_secim)
else:
    # --- GİRİŞ YAPILMIŞ MOD ---
    with st.sidebar:
        st.success(f"Giriş Yapıldı: {st.session_state.username}")
        if st.button("Oturumu Kapat"):
            st.session_state.username = None
            st.rerun()
    
    data_result = load_dynamic_data(st.session_state.username, st.session_state.token)

# --- 5. ANALİZ VE GÖRSELLEŞTİRME (EN STABİL HALİ) ---
if isinstance(data_result, pd.DataFrame):
    # Düğüm ve Kenar Oluşturma
    G = nx.from_pandas_edgelist(data_result, source=data_result.columns[0], target=data_result.columns[1])
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    metrics_df = pd.DataFrame({
        'node': list(degree_cent.keys()),
        'degree': list(degree_cent.values()),
        'betweenness': list(betweenness.values())
    })

    # KNN Analizi
    if len(metrics_df) > 3:
        try:
            X = metrics_df[['degree', 'betweenness']].values
            y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
            X_scaled = StandardScaler().fit_transform(X)
            knn = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1)).fit(X_scaled, y)
            metrics_df['AI_Kategori'] = knn.predict(X_scaled)
            metrics_df['color'] = metrics_df['AI_Kategori'].map({1: "#e74c3c", 0: "#3498db"})
            metrics_df['Durum'] = metrics_df['AI_Kategori'].map({1: "🎯 Stratejik", 0: "👤 Normal"})
        except:
            metrics_df['color'] = "#3498db"
            metrics_df['Durum'] = "Normal"

    # Pyvis Grafik Çizimi
    st.subheader("🌐 İnteraktif Ağ Haritası")
    try:
        net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="black")
        for _, row in metrics_df.iterrows():
            net.add_node(row['node'], label=str(row['node']), color=row.get('color', "#3498db"))
        for edge in G.edges():
            net.add_edge(edge[0], edge[1])
        net.toggle_physics(True)
        html_data = net.generate_html()
        components.html(html_data, height=600, scrolling=False)
    except:
        st.error("Grafik oluşturulurken bir sorun oluştu.")

    # Tablo ve İndirme
    st.divider()
    st.dataframe(metrics_df[['node', 'degree', 'betweenness', 'Durum']], use_container_width=True)
    
    csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📄 Analiz Raporunu İndir (CSV)", csv_data, f"rapor_{st.session_state.username}.csv")

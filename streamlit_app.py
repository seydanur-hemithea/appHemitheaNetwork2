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

if isinstance(data_result, str):
    # Hata durumlarını yönet
    if data_result == "CONNECTION_ERROR":
        st.error("📡 Sunucu uyanıyor... Lütfen 10 saniye bekleyip tekrar deneyiniz.")
        if st.button("Tekrar Bağlan"): st.rerun()
    elif data_result == "NOT_FOUND":
        st.info(f"🔍 Hoş geldin {current_user}! Veri bulunamadı.")
        if st.button("🔄 Veriyi Kontrol Et"): st.rerun()
    elif data_result == "EMPTY":
        st.warning("⚠️ Dosya bulundu ancak içeriği boş!")
    else:
        st.error(f"📡 Beklenmedik durum: {data_result}")
    st.stop()

elif isinstance(data_result, pd.DataFrame):
    st.success(f"✅ Hoş geldin {current_user}")
    
    # 1. TEMEL AĞ ANALİZİ (G Nesnesi)
    G = nx.from_pandas_edgelist(data_result, source=data_result.columns[0], target=data_result.columns[1])
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    metrics_df = pd.DataFrame({
        'node': list(degree_cent.keys()),
        'degree': list(degree_cent.values()),
        'betweenness': list(betweenness.values())
    })

    # 2. KNN (YAPAY ZEKA) MOTORU
    if len(metrics_df) > 3:
        try:
            X = metrics_df[['degree', 'betweenness']].values
            y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            knn = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1))
            knn.fit(X_scaled, y)
            
            metrics_df['AI_Kategori'] = knn.predict(X_scaled)
            metrics_df['Rol_Tanimi'] = metrics_df['AI_Kategori'].map({
                1: "🎯 Stratejik Köprü", 
                0: "👤 Normal Aktör"
            })
            metrics_df['color'] = metrics_df['AI_Kategori'].map({1: "#e74c3c", 0: "#3498db"})
        except Exception as e:
            st.error(f"KNN Hatası: {e}")

    # 3. GÖRSELLEŞTİRME VE TABLAR (Android Optimize)
    tab1, tab2 = st.tabs(["🕸️ Ağ Haritası", "📊 Analiz Raporu"])

    with tab1:
        st.subheader("🌐 Ağ Yapısı")
        # Android'de kararma yapmaması için sabit resim (Matplotlib) kullanıyoruz
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 7))
            pos = nx.spring_layout(G, k=0.6)
            # KNN renklerini kullanarak çiz
            colors = [metrics_df[metrics_df['node'] == n]['color'].values[0] if 'color' in metrics_df.columns else '#3498db' for n in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=colors, node_size=600, font_size=7, edge_color='#ecf0f1')
            st.pyplot(fig)
            plt.close(fig)
        except: st.write("Harita yüklenirken bir sorun oluştu.")

    with tab2:
        st.subheader("🤖 YZ Analitik Raporlama")
        st.dataframe(metrics_df[['node', 'degree', 'betweenness', 'Rol_Tanimi']], width="stretch")
        
        st.divider()
        st.write("📂 **Analiz Çıktılarını İndir**")
        c1, c2 = st.columns(2)
        with c1:
            csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📄 Veri Raporu (CSV)", csv_data, f"analiz_{current_user}.csv")
        with c2:
            try:
                # İndirilen dosya interaktif (Pyvis) olacak
                net_dl = Network(height="700px", width="100%", bgcolor="#ffffff")
                net_dl.from_nx(G)
                # Buradaki HTML'de grafiğin görünmesi için CDN ayarını manuel yapmıyoruz (Pyvis hallediyor)
                temp_path = os.path.join(tempfile.gettempdir(), f"final_{current_user}.html")
                net_dl.save_graph(temp_path)
                with open(temp_path, 'r', encoding='utf-8') as f:
                    html_string = f.read()
                st.download_button("🌐 İnteraktif Ağ (HTML)", html_string, f"ag_{current_user}.html", "text/html")
            except: st.write("Hazırlanıyor...") 
        with c3:
            plt.clf()
            fig_tbl, ax_tbl = plt.subplots(figsize=(10, 6))
            ax_tbl.axis('off')
            ax_tbl.table(cellText=metrics_df.head(10).values, colLabels=metrics_df.columns, loc='center')
            buf_tbl = BytesIO()
            plt.savefig(buf_tbl, format="png")
            st.download_button("🖼️ Tablo Resmi", buf_tbl.getvalue(), "tablo.png", "image/png")
        with c4:
            plt.clf()
            fig_gr, ax_gr = plt.subplots(figsize=(10, 8))
            nx.draw(G, with_labels=True, node_color='#3498db')
            buf_gr = BytesIO()
            plt.savefig(buf_gr, format="png")
            st.download_button("📸 Ağ Resmi", buf_gr.getvalue(), "ag.png", "image/png")

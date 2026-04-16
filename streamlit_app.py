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
import streamlit.components.v1 as components
import plotly.graph_objects as go

# --- 1. GITHUB RAW ÇEVİRİCİ ALGORİTMA ---
def to_raw(url):
    if "github.com" in url and "raw" not in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

# --- 2. VİTRİN VERİ FONKSİYONU ---
def get_vitrin_data(secim):
    linkler = {
        "Efendi Analizi": "https://github.com/seydanur-hemithea/appHemitheaNetwork2/blob/main/Efendi.csv",
        "Game of Thrones": "https://github.com/seydanur-hemithea/appHemitheaNetwork2/blob/main/GoT.csv"
    }
    try:
        raw_url = to_raw(linkler[secim])
        response = requests.get(raw_url)
        df = pd.read_csv(StringIO(response.text))
        return df
    except:
        return pd.DataFrame({'Kaynak': ['Hata'], 'Hedef': ['Veri Alınamadı']})

# --- 3. OTURUM YÖNETİMİ ---
if "username" not in st.session_state:
    st.session_state.username = None
if "login_mode" not in st.session_state:
    st.session_state.login_mode = False

# --- 4. VERİ ÇEKME (RENDER) ---
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

# --- 5. ANA EKRAN TASARIMI ---
st.set_page_config(page_title="Hemithea Network", layout="wide")
st.title("🌐 Hemithea Network Analytics")

# SIDEBAR: Giriş ve Kontrol
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100) # Şirin bir ikon
    st.subheader("🛠️ Kontrol Paneli")
    
    if not st.session_state.username:
        if st.button("🚀 Kendi Verini Analiz Et (Giriş Yap)"):
            st.session_state.login_mode = True
        
        if st.session_state.login_mode:
            st.divider()
            u_id = st.text_input("Kullanıcı Kimliği:")
            u_tk = st.text_input("Erişim Tokenı:", type="password")
            if st.button("Sistemi Başlat"):
                st.session_state.username = u_id
                st.session_state.token = u_tk
                st.session_state.login_mode = False
                st.rerun()
            if st.button("Vitrine Dön"):
                st.session_state.login_mode = False
                st.rerun()
    else:
        st.success(f"Oturum Açıldı: {st.session_state.username}")
        if st.button("🚪 Çıkış Yap"):
            st.session_state.username = None
            st.rerun()

# VERİ SEÇİMİ
if not st.session_state.username:
    st.info("👋 **Hemithea Portfolyo Modu.** Örnek analizleri inceleyin veya kendi verileriniz için giriş yapın.")
    secilen_analiz = st.radio("İncelemek istediğiniz portfolyo çalışması:", 
                              ["Efendi Analizi", "Game of Thrones"], horizontal=True)
    data_result = get_vitrin_data(secilen_analiz)
else:
    data_result = load_dynamic_data(st.session_state.username, st.session_state.token)

# --- 6. ANALİZ VE GÖRSELLEŞTİRME ---
if isinstance(data_result, pd.DataFrame):
    # Grafik Hazırlığı
    G = nx.from_pandas_edgelist(data_result, source=data_result.columns[0], target=data_result.columns[1])
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    metrics_df = pd.DataFrame({
        'node': list(degree_cent.keys()),
        'degree': list(degree_cent.values()),
        'betweenness': list(betweenness.values())
    })

    tab1, tab2 = st.tabs(["🕸️ Analiz Haritası", "📊 YZ Metrikleri"])

    with tab1:
        if not st.session_state.username:
            # VİTRİN MODU: SEVDİĞİN PYVIS YAPISI
            st.subheader(f"✨ {secilen_analiz} - Hareketli Ağ")
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            net.from_nx(G)
            net.toggle_physics(True)
            try:
                html_data = net.generate_html()
                components.html(html_data, height=650, scrolling=True)
            except: st.write("Grafik yükleniyor...")
        else:
            # PROFESYONEL MOD: STABİL PLOTLY
            st.subheader("📊 Kişisel Analiz Paneliniz")
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.8, color='#888'), hoverinfo='none', mode='lines')
            
            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]; node_x.append(x); node_y.append(y)
                deg = degree_cent.get(node, 0)
                node_text.append(f"Düğüm: {node}<br>Derece: {deg:.2f}")
                node_color.append(deg); node_size.append(max(15, 20 + (deg * 50)))
            
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text', text=[str(n) for n in G.nodes()],
                textposition="top center", hoverinfo='text', hovertext=node_text,
                marker=dict(showscale=True, colorscale='Viridis', color=node_color, size=node_size, line=dict(width=2, color='white'))
            )
            
            fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, hovermode='closest', margin=dict(b=0, l=0, r=0, t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # KNN Analizi ve Tablo (Eski kodun aynen korundu)
        st.subheader("🤖 Yapay Zeka (KNN) Raporu")
        if len(metrics_df) > 3:
            try:
                X_rep = metrics_df[['degree', 'betweenness']].values
                X_rep_scaled = StandardScaler().fit_transform(X_rep)
                y_rep = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
                knn_rep = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1)).fit(X_rep_scaled, y_rep)
                metrics_df['Rol_Tanimi'] = knn_rep.predict(X_rep_scaled)
                metrics_df['Durum'] = metrics_df['Rol_Tanimi'].map({1: "Stratejik Köprü", 0: "Normal Aktör"})
                st.info("💡 Aktörler ağdaki konumlarına göre sınıflandırıldı.")
            except: pass
        
        st.dataframe(metrics_df, use_container_width=True)
        st.divider()
        
        # İndirme Butonları (Senin kolon yapın)
        st.write("📂 **Dosyaları İndir**")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📄 CSV Raporu", metrics_df.to_csv(index=False).encode('utf-8-sig'), "rapor.csv", "text/csv")
        with c2:
            try:
                net_dl = Network(height="600px", width="100%")
                net_dl.from_nx(G)
                temp_p = os.path.join(tempfile.gettempdir(), "ag.html")
                net_dl.save_graph(temp_p)
                with open(temp_p, 'r', encoding='utf-8') as f: html_s = f.read()
                st.download_button("🌐 HTML Analiz", html_s, "analiz.html", "text/html")
            except: st.write("Hazırlanıyor...")

        st.write("📸 **Görsel Kayıtlar**")
        c3, c4 = st.columns(2)
        with c3:
            plt.clf(); fig_t, ax_t = plt.subplots(figsize=(10, 6)); ax_t.axis('off')
            ax_t.table(cellText=metrics_df.head(15).values, colLabels=metrics_df.columns, loc='center')
            buf_t = BytesIO(); plt.savefig(buf_t, format="png"); st.download_button("🖼️ Tablo PNG", buf_t.getvalue(), "tablo.png", "image/png")
        with c4:
            plt.clf(); fig_g, ax_g = plt.subplots(figsize=(10, 8))
            nx.draw(G, with_labels=True, node_color='#3498db', node_size=300); buf_g = BytesIO(); plt.savefig(buf_g, format="png")
            st.download_button("📸 Ağ PNG", buf_g.getvalue(), "ag.png", "image/png")

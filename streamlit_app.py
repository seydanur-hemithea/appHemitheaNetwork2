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

# --- 1. OTURUM VE PARAMETRE YÖNETİMİ ---
if "username" not in st.session_state:
    st.session_state.username = None

# query_params artık yeni Streamlit versiyonlarında bu şekilde alınıyor
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
        # Senin Render üzerindeki yükleme klasörün
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

# --- 4. DURUM KONTROLLERİ ---
if isinstance(data_result, str):
    if data_result == "CONNECTION_ERROR":
        st.error("📡 Sunucu uyanıyor... Lütfen 10 saniye bekleyip tekrar deneyiniz.")
    elif data_result == "NOT_FOUND":
        st.info(f"🔍 Hoş geldin {current_user}! Henüz analiz edilecek veri yüklenmemiş.")
    elif data_result == "EMPTY":
        st.warning("⚠️ Dosya bulundu ancak içeriği boş!")
    else:
        st.error("📡 Sunucu ile bağlantı kurulamıyor.")
    st.stop()

elif isinstance(data_result, pd.DataFrame):
    st.success(f"✅ Hoş geldin {current_user}")
    
    # --- ANALİZLER ---
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
        st.subheader("🌐 Etkileşimli Analiz Paneli")
        
        # 1. Koordinatları Hesapla (Spring Layout)
        # Bu, düğümleri dengeli bir şekilde ekrana yayar
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 2. Kenarları (Çizgileri) Hazırla
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
    
        # 3. Düğümleri (Noktaları) Hazırla
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
    
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Üzerine gelince çıkacak olan isim ve veriler (Hover)
            deg = degree_cent.get(node, 0)
            bet = betweenness.get(node, 0)
            node_text.append(f"<b>Düğüm:</b> {node}<br><b>Derece:</b> {deg:.2f}<br><b>Arasındalık:</b> {bet:.2f}")
            
            # Renklendirme ve Boyutlandırma
            node_color.append(deg) # Dereceye göre renk değişsin
            node_size.append(20 + (deg * 50)) # Dereceye göre boyut büyüsün
    
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[str(n) for n in G.nodes()], # İsimler her zaman görünsün istiyorsan kalsın
            textposition="top center",
            hoverinfo='text',
            hovertext=node_text, # Üzerine gelince detaylar çıkar
            marker=dict(
                showscale=True,
                colorscale='Viridis', # Şirin ve profesyonel bir renk skalası
                reversescale=True,
                color=node_color,
                size=node_size,
                colorbar=dict(thickness=15, title='Etki Seviyesi', xanchor='left', titleside='right'),
                line_width=2)
        )
    
        # 4. Figürü Oluştur
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
        # 5. Ekrana Bas (Android Dostu Komut)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("🤖 Yapay Zeka (KNN) Raporu")
        if len(metrics_df) > 3:
            try:
                # KNN analizini burada da yapıyoruz ki tabloda görünsün
                X_rep = metrics_df[['degree', 'betweenness']].values
                X_rep_scaled = StandardScaler().fit_transform(X_rep)
                y_rep = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
                knn_rep = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1)).fit(X_rep_scaled, y_rep)
                metrics_df['Rol_Tanimi'] = knn_rep.predict(X_rep_scaled)
                metrics_df['Durum'] = metrics_df['Rol_Tanimi'].map({1: "Stratejik Köprü", 0: "Normal Aktör"})
                st.info("💡 Aktörler ağdaki stratejik konumlarına göre sınıflandırıldı.")
            except: pass
        
        st.dataframe(metrics_df, use_container_width=True)
        st.divider()
        
        st.write("📂 **Analiz Çıktılarını İndir**")
        c1, c2 = st.columns(2)
        with c1:
            csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📄 Veri Raporu (CSV)", csv_data, f"rapor_{current_user}.csv", "text/csv")
        with c2:
            try:
                net_dl = Network(height="600px", width="100%", bgcolor="#ffffff")
                net_dl.from_nx(G)
                temp_path = os.path.join(tempfile.gettempdir(), f"dl_{current_user}.html")
                net_dl.save_graph(temp_path)
                with open(temp_path, 'r', encoding='utf-8') as f:
                    html_s = f.read()
                st.download_button("🌐 İnteraktif Ağ (HTML)", html_s, f"ag_{current_user}.html", "text/html")
            except: st.write("Hazırlanıyor...")

        st.write("📸 **Görsel Kayıtlar**")
        c3, c4 = st.columns(2) 
        with c3:
            plt.clf()
            fig_tbl, ax_tbl = plt.subplots(figsize=(10, 6))
            ax_tbl.axis('off')
            # Font hatası almamak için metriklerin ilk 15 satırını çiziyoruz
            ax_tbl.table(cellText=metrics_df.head(15).values, colLabels=metrics_df.columns, loc='center')
            buf_tbl = BytesIO()
            plt.savefig(buf_tbl, format="png", dpi=100)
            plt.close(fig_tbl) # Belleği boşalt
            st.download_button("🖼️ Tablo Resmi", buf_tbl.getvalue(), "tablo.png", "image/png")
        with c4:
            plt.clf()
            fig_gr, ax_gr = plt.subplots(figsize=(10, 8))
            nx.draw(G, with_labels=True, node_color='#3498db', node_size=300, font_size=7)
            buf_gr = BytesIO()
            plt.savefig(buf_gr, format="png", dpi=100)
            plt.close(fig_gr) # Belleği boşalt
            st.download_button("📸 Ağ Resmi", buf_gr.getvalue(), "ag.png", "image/png")

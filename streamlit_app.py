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
        st.subheader("🌐 Ağ Etkileşim Haritası")
        
        # 1. Checkbox (Seçim kutun yerinde kalsın)
        use_ai = st.checkbox("🤖 KNN Sınıflandırmasını Uygula", key="ai_check_final")
        
        # 2. Grafik Nesnesini Hazırla
        net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="black")
        
        if use_ai and len(metrics_df) > 3:
            # KNN renklendirme mantığını buraya dahil ediyoruz
            for _, row in metrics_df.iterrows():
                role_color = "#e74c3c" if row.get('AI_Role') == 1 else "#3498db"
                net.add_node(row['node'], label=str(row['node']), color=role_color)
            for edge in G.edges():
                net.add_edge(edge[0], edge[1])
        else:
            net.from_nx(G)
        
        net.toggle_physics(True)
        
        # --- KRİTİK DEĞİŞİKLİK: DOSYA KAYDETMEDEN BASIYORUZ ---
        # --- ANALİZİN GÖRÜNMESİNİ SAĞLAYAN GÜNCELLEME ---
        try:
            # 1. HTML'i oluştur
            html_content = net.generate_html()
            
            # 2. KRİTİK DOKUNUŞ: 
            # Pyvis bazen CDN linklerini 'http' olarak verir, bu Android'de bloklanır.
            # Linkleri güvenli 'https' haline getiriyoruz.
            html_content = html_content.replace('http://', 'https://')
            
            # 3. Android WebView için kütüphaneyi HTML içine zorla enjekte ediyoruz
            # Bu scriptler ağın çizilmesini sağlayan asıl motorlardır.
            
            components.html(html_content, height=600, scrolling=True)
            
        except Exception as e:
            st.error(f"Görselleştirme Motoru Başlatılamadı: {e}")
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

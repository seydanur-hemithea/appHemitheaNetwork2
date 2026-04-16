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

# --- 1. OTURUM VE PARAMETRE YÖNETİMİ ---
# Parametreleri al ve session_state'e aktar
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
        # Timeout'u 20 saniyeye çıkardık (Render uyanana kadar beklesin)
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

# Eğer kullanıcı adı yoksa ekranı kilitle
if not current_user:
    st.warning("🔑 Giriş bilgileri bekleniyor... Android üzerinden yönlendirme gerekiyor.")
    # Manuel deneme butonu
    if st.button("🔄 Oturumu Yenile"):
        st.rerun()
    st.stop()

data_result = load_dynamic_data(current_user, current_token)

# --- 4. DURUM KONTROLLERİ ---
if data_result == "CONNECTION_ERROR":
    st.error("📡 Sunucu uyanıyor... Lütfen bağlantınızı kontrol edin ve 10 saniye bekleyip tekrar deneyin.")
    if st.button("Tekrar Bağlan"):
        st.rerun()
    st.stop()

elif isinstance(data_result, pd.DataFrame):
    st.success(f"✅ Hoş geldin {current_user}")


    tab1, tab2 = st.tabs(["🕸️ Analiz Haritası", "📊 YZ Metrikleri"])

    with tab1:
        use_ai = st.checkbox("🤖 KNN Sınıflandırmasını Uygula")
        net = Network(height="500px", width="100%", bgcolor="#ffffff")
        
        if use_ai and len(metrics_df) > 3:
            X = metrics_df[['degree', 'betweenness']].values
            y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            knn = KNeighborsClassifier(n_neighbors=min(3, len(X)-1))
            knn.fit(X_scaled, y)
            metrics_df['AI_Role'] = knn.predict(X_scaled)
            
            for _, row in metrics_df.iterrows():
                color = "#e74c3c" if row['AI_Role'] == 1 else "#3498db"
                net.add_node(row['node'], label=row['node'], color=color)
            net.from_nx(G)
        else:
            net.from_nx(G)
        
        net.toggle_physics(True)
        components.html(net.generate_html(), height=550)

    with tab2:
        st.subheader("🤖 Yapay Zeka (KNN) ve Analitik Raporlama")
        
        if len(metrics_df) > 3:
            try:
                X = metrics_df[['degree', 'betweenness']].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
                knn = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1))
                knn.fit(X_scaled, y)
                metrics_df['AI_Kategori'] = knn.predict(X_scaled)
                metrics_df['Rol_Tanimi'] = metrics_df['AI_Kategori'].map({1: "Stratejik Köprü", 0: "Normal Aktör"})
                st.info("💡 KNN Modeli: Aktörler ağdaki stratejik konumlarına göre sınıflandırıldı.")
            except Exception as e:
                st.error(f"YZ Analizi yapılamadı: {e}")
        
        st.dataframe(metrics_df, use_container_width=True)
        st.divider()
        st.write("📂 **Analiz Çıktılarını İndir**")
        
        # Dosyalar ve Etkileşimli Ağ
        c1, c2 = st.columns(2)
        with c1:
            csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📄 Veri Raporu (CSV)", csv_data, f"analiz_{current_user}.csv", "text/csv")
        with c2:
            try:
                net_dl = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
                net_dl.from_nx(G)
                net_dl.toggle_physics(True)
                net_dl.set_cdn_resources(True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    net_dl.save_graph(tmp.name)
                    with open(tmp.name, 'r', encoding='utf-8') as f:
                        html_string = f.read()
                st.download_button("🌐 Etkileşimli Ağ (HTML)", html_string, f"interaktif_ag_{current_user}.html", "text/html")
            except: st.write("HTML Dosyası Hazırlanıyor...")

        st.write("📸 **Görsel Kayıtlar**")
        # --- DÜZELTİLEN KISIM: c3 ve c4 kolonları burada tanımlandı ---
        c3, c4 = st.columns(2) 
        with c3:
            try:
                plt.clf()
                fig_tbl, ax_tbl = plt.subplots(figsize=(10, 6))
                ax_tbl.axis('off')
                the_table = ax_tbl.table(cellText=metrics_df.head(20).values, colLabels=metrics_df.columns, loc='center')
                buf_tbl = BytesIO()
                plt.savefig(buf_tbl, format="png", dpi=150, bbox_inches='tight')
                plt.close(fig_tbl)
                st.download_button("🖼️ Tablo Resmi", buf_tbl.getvalue(), "tablo.png", "image/png")
            except: st.write("Hazırlanıyor...")
        with c4:
            try:
                plt.clf()
                fig_gr, ax_gr = plt.subplots(figsize=(10, 8))
                nx.draw(G, nx.spring_layout(G, k=0.5), with_labels=True, node_color='#3498db', node_size=400, font_size=7)
                buf_gr = BytesIO()
                plt.savefig(buf_gr, format="png", dpi=150, bbox_inches='tight')
                plt.close(fig_gr)
                st.download_button("📸 Ağ Resmi", buf_gr.getvalue(), "ag.png", "image/png")
            except: st.write("Hazırlanıyor...")

elif data_result == "NOT_FOUND":
    st.info(f"🔍 Hoş geldin {current_user}! Henüz analiz edilecek bir verin yüklü değil.")
    with st.expander("❓ Verimi Nasıl Yüklerim?", expanded=True):
        st.markdown("1. Uygulamadan veri yükleme ekranına git.\n2. CSV dosyanı seç.\n3. Yükleme bitince bu sayfayı yenile.")
    if st.button("🔄 Veriyi Şimdi Kontrol Et"):
        st.rerun()

elif data_result == "EMPTY":
    st.warning("⚠️ Dosya bulundu ancak içeriği boş!")

else:
    st.error("📡 Sunucu ile bağlantı kurulamıyor.")

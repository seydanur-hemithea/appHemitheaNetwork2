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

# --- 1. OTURUM VE PARAMETRE YÖNETİMİ ---
# Parametreleri her zaman kontrol et ve session_state'e aktar
query_params = st.query_params
if "username" in query_params:
    st.session_state.username = query_params["username"]
    st.session_state.token = query_params.get("token", "")

# --- 2. GÜVENLİ VERİ ÇEKME ---
@st.cache_data(ttl=2)
def load_dynamic_data(uname, token):
    if not uname or not token:
        return None
    try:
        target_url = f"https://apphemitheanetwork.onrender.com/uploads/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=12)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df if not df.empty else "EMPTY"
        return "NOT_FOUND"
    except:
        return "ERROR"

# --- 3. ANA AKIŞ ---
st.title("🌐 Hemithea Network Analytics")

# Session state'den bilgileri güvenli bir değişkene alalım
current_user = st.session_state.get("username")
current_token = st.session_state.get("token")

if not current_user:
    st.warning("🔑 Giriş bilgileri bekleniyor... Lütfen Android üzerinden veriyi yükleyin.")
    # Eğer parametreler geç gelirse diye sayfayı yenileme butonu
    if st.button("Sistemi Yenile"):
        st.rerun()
    st.stop() # Kullanıcı yoksa aşağıya geçme (kararmayı önler)

data_result = load_dynamic_data(current_user, current_token)

if isinstance(data_result, pd.DataFrame):
    st.success(f"✅ Hoş geldin {current_user}")
    
    # --- 4. ANALİZ MOTORU ---
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
        st.subheader("🤖 KNN Tahmin Sonuçları")
        st.dataframe(metrics_df, use_container_width=True)
        
        st.divider()
        st.write("📂 **Dosyaları İndir**")
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel uyumlu CSV (utf-8-sig)
            csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📄 Analizi İndir (CSV)",
                data=csv_data,
                file_name=f"hemithea_metrics_{current_user}.csv",
                mime="text/csv"
            )
        
        with col2:
            # PNG İndirme
            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 6))
            nx.draw(G, with_labels=True, node_color='skyblue', node_size=600, font_size=8)
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            st.download_button(
                label="📸 Ağı İndir (PNG)",
                data=buf.getvalue(),
                file_name=f"hemithea_graph_{current_user}.png",
                mime="image/png"
            )

elif data_result == "NOT_FOUND":
    st.info("🔍 Analiz edilecek veri henüz hazır değil.")
else:
    st.error("📡 Sunucu bağlantı hatası.")

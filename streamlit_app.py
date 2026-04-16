import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
from io import StringIO
from sklearn.neighbors import KNeighborsClassifier

# --- 1. OTURUM HAFIZASI (SESSION STATE) ---
# Sayfa her yenilendiğinde parametrelerin kaybolmasını engeller
if 'username' not in st.session_state:
    params = st.query_params
    st.session_state.username = params.get("username")
    st.session_state.token = params.get("token")

# --- 2. GÜVENLİ VERİ ÇEKME ---
@st.cache_data(ttl=5)
def load_dynamic_data(uname, token):
    if not uname or not token:
        return None
    try:
        target_url = f"https://apphemitheanetwork.onrender.com/uploads/{uname}/network_data.csv?token={token}"
        # Akış hatasını önlemek için stream=True kullanabiliriz
        response = requests.get(target_url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df if not df.empty else "EMPTY"
        return "NOT_FOUND"
    except:
        return "ERROR"

# --- 3. ANA AKIŞ ---
st.title("🌐 Hemithea Network Analytics")

# Parametreler session_state'den okunur (Kararmayı önleyen kritik nokta)
uname = st.session_state.username
token = st.session_state.token

if not uname or not token:
    st.warning("🔑 Giriş bilgileri bekleniyor... Eğer uygulama içinden açtıysanız lütfen bekleyin.")
    # Debug için (sadece sen gör): st.write(f"Parametreler: {st.query_params}")
else:
    data_result = load_dynamic_data(uname, token)
    
    if isinstance(data_result, pd.DataFrame):
        st.success(f"✅ Oturum Açıldı: {uname}")
        
        # Grafik ve Metrik Hazırlığı
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
            # KNN Renklendirme
            use_ai = st.checkbox("🤖 YZ (KNN) ile Analiz Et")
            net = Network(height="500px", width="100%", bgcolor="#ffffff")
            
            if use_ai and len(metrics_df) > 3:
                X = metrics_df[['degree', 'betweenness']].values
                y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
                knn = KNeighborsClassifier(n_neighbors=min(3, len(X)-1))
                knn.fit(X, y)
                metrics_df['AI_Role'] = knn.predict(X)
                
                for _, row in metrics_df.iterrows():
                    c = "red" if row['AI_Role'] == 1 else "skyblue"
                    net.add_node(row['node'], label=row['node'], color=c)
                net.from_nx(G)
            else:
                net.from_nx(G)
            
            net.toggle_physics(True)
            components.html(net.generate_html(), height=550)

        with tab2:
            st.dataframe(metrics_df, use_container_width=True)

    elif data_result == "NOT_FOUND":
        st.info("🔍 Veri bekleniyor... Lütfen uygulamadan dosya yükleyin.")
    else:
        st.error("📡 Bağlantı hatası veya dosya henüz hazır değil.")
        with tab3:
            st.dataframe(data_result, use_container_width=True)

    elif data_result == "EMPTY":
        st.info("📂 Dosya boş.")
    elif data_result == "NOT_FOUND":
        st.info("🔍 Veri yüklenmesi bekleniyor...")
    elif isinstance(data_result, str) and "ERROR" in data_result:
        st.error(f"📡 Bağlantı hatası: {data_result}")

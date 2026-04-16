import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
from io import StringIO, BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. Parametre Kontrolü ---
params = st.query_params
current_username = params.get("username")
current_token = params.get("token")

# --- 2. Güvenli Veri Çekme Fonksiyonu ---
@st.cache_data(ttl=1)
def load_dynamic_data(uname, token):
    if not uname or not token:
        return None
    try:
        target_url = f"https://apphemitheanetwork.onrender.com/uploads/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df if not df.empty else "EMPTY"
        return "NOT_FOUND"
    except Exception as e:
        return f"ERROR: {str(e)}"

# --- 3. Ana Arayüz ---
st.title("🌐 Hemithea Network Analytics")

if not current_username or not current_token:
    st.warning("🔑 Lütfen uygulama üzerinden giriş yapın...")
else:
    data_result = load_dynamic_data(current_username, current_token)
    
    if isinstance(data_result, pd.DataFrame):
        st.success(f"✅ Veri Hazır: {current_username}")
        
        # Grafik Objesi
        G = nx.from_pandas_edgelist(data_result, source=data_result.columns[0], target=data_result.columns[1])
        
        # Metrikleri en başta hesaplayalım ki her yerde kullanalım
        degree_cent = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        metrics_df = pd.DataFrame({
            'node': list(degree_cent.keys()),
            'degree': list(degree_cent.values()),
            'betweenness': list(betweenness.values())
        })

        tab1, tab2, tab3 = st.tabs(["🕸️ Analiz Haritası", "📊 YZ Metrikleri", "📄 Ham Veri"])
        
        with tab1:
            st.subheader("Ağ Görselleştirmesi")
            
            # KNN ve Renklendirme Kontrolü
            use_ai = st.checkbox("🤖 YZ ile Stratejik Aktörleri Renklendir (KNN)")
            
            net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
            
            if use_ai and len(metrics_df) > 3:
                # KNN Hazırlık
                X = metrics_df[['degree', 'betweenness']].values
                avg_bet = metrics_df['betweenness'].mean()
                y = (metrics_df['betweenness'] > avg_bet).astype(int)
                
                knn = KNeighborsClassifier(n_neighbors=min(3, len(X)-1))
                knn.fit(X, y)
                metrics_df['AI_Role'] = knn.predict(X)
                
                # Renkli Düğümler
                for _, row in metrics_df.iterrows():
                    color = "red" if row['AI_Role'] == 1 else "skyblue"
                    label = f"{row['node']} ({'Stratejik' if row['AI_Role'] == 1 else 'Normal'})"
                    net.add_node(row['node'], label=row['node'], color=color, title=label)
                net.from_nx(G) # Bağlantıları ekle
            else:
                net.from_nx(G) # Standart Çizim
            
            net.toggle_physics(True)
            components.html(net.generate_html(), height=550)

        with tab2:
            st.subheader("🤖 KNN Gruplandırma Analizi")
            st.dataframe(metrics_df, use_container_width=True)
            
            # CSV İndirme
            csv = metrics_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📄 Analizi İndir", csv, "ai_metrics.csv", "text/csv")

        with tab3:
            st.dataframe(data_result, use_container_width=True)

    elif data_result == "EMPTY":
        st.info("📂 Dosya boş.")
    elif data_result == "NOT_FOUND":
        st.info("🔍 Veri yüklenmesi bekleniyor...")
    elif isinstance(data_result, str) and "ERROR" in data_result:
        st.error(f"📡 Bağlantı hatası: {data_result}")

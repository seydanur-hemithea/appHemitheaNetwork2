import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
from io import StringIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler # YZ için kritik

# --- 1. OTURUM HAFIZASI (SESSION STATE) ---
# Eğer boşsa veya yeni parametre geldiyse güncelle
params = st.query_params
if "username" not in st.session_state or (params.get("username") and params.get("username") != st.session_state.username):
    st.session_state.username = params.get("username")
    st.session_state.token = params.get("token")

# --- 2. GÜVENLİ VERİ ÇEKME ---
@st.cache_data(ttl=3) # TTL'i 3 saniye yaptık (Hızlı güncelleme için)
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

uname = st.session_state.username
token = st.session_state.token

if not uname or not token:
    st.warning("🔑 Giriş bilgileri bekleniyor... Android üzerinden tekrar giriş yapmayı deneyin.")
else:
    data_result = load_dynamic_data(uname, token)
    
    if isinstance(data_result, pd.DataFrame):
        st.success(f"✅ Oturum Aktif: {uname}")
        
        # 4. ANALİZ MOTORU
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
            net = Network(height="550px", width="100%", bgcolor="#ffffff")
            
            if use_ai and len(metrics_df) > 3:
                # --- YZ STANDARTLARINA UYGUN KNN ---
                X = metrics_df[['degree', 'betweenness']].values
                y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
                
                # Veriyi ölçeklendir (Scale)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                knn = KNeighborsClassifier(n_neighbors=min(3, len(X)-1))
                knn.fit(X_scaled, y)
                metrics_df['AI_Role'] = knn.predict(X_scaled)
                
                # Renklendirilmiş Düğümler
                for _, row in metrics_df.iterrows():
                    color = "#e74c3c" if row['AI_Role'] == 1 else "#3498db" # Kırmızı vs Mavi
                    title = f"Rol: {'Stratejik' if row['AI_Role'] == 1 else 'Normal'}"
                    net.add_node(row['node'], label=row['node'], color=color, title=title)
                
                # Kenarları ekle (Bağlantıların kaybolmaması için)
                for edge in G.edges():
                    net.add_edge(edge[0], edge[1])
            else:
                net.from_nx(G)
            
            net.toggle_physics(True)
            components.html(net.generate_html(), height=600)

        with tab2:
            st.subheader("🤖 KNN Tahmin Sonuçları")
            st.dataframe(metrics_df, use_container_width=True)
            
            st.divider()
            st.write("📂 *Dosyaları İndir*")
            
            # Yan yana iki kolon oluşturalım
            col1, col2 = st.columns(2)
            
            with col1:
                # --- CSV İNDİRME (Excel Uyumlu) ---
                csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📄 Metrikleri İndir (CSV)",
                    data=csv_data,
                    file_name=f"hemithea_analiz_{u}.csv",
                    mime="text/csv",
                    key="download-csv"
                )
            
            with col2:
                # --- PNG İNDİRME (Ağ Grafiği) ---
                try:
                    plt.clf()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    pos = nx.spring_layout(G)
                    nx.draw(G, pos, ax=ax, with_labels=True, 
                            node_color='skyblue', node_size=700, 
                            width=1.0, font_size=8)
                    
                    buf = BytesIO()
                    plt.savefig(buf, format="png", dpi=150)
                    plt.close(fig) # Belleği temizle
                    
                    st.download_button(
                        label="📸 Ağ Grafiğini İndir (PNG)",
                        data=buf.getvalue(),
                        file_name=f"hemithea_graph_{u}.png",
                        mime="image/png",
                        key="download-png"
                    )
                except Exception as e:
                    st.error(f"Grafik hazırlanamadı: {e}")

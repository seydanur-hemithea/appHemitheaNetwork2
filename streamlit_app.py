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

# --- 4. ULTRA HAFİF AKIŞ (ANDROID ÇÖZÜMÜ) ---

if isinstance(data_result, pd.DataFrame):
    st.success(f"✅ Bağlantı Başarılı: {current_user}")
    
    # 1. HESAPLAMALAR (Sadece gerekli olanlar)
    G = nx.from_pandas_edgelist(data_result, source=data_result.columns[0], target=data_result.columns[1])
    degree_cent = nx.degree_centrality(G)
    
    metrics_df = pd.DataFrame({
        'Aktör': list(degree_cent.keys()),
        'Etki_Skoru': list(degree_cent.values())
    }).sort_values(by='Etki_Skoru', ascending=False)

    # 2. KNN ANALİZİ (Basitleştirilmiş)
    if len(metrics_df) > 2:
        threshold = metrics_df['Etki_Skoru'].mean()
        metrics_df['Rol'] = metrics_df['Etki_Skoru'].apply(lambda x: "🎯 Stratejik" if x > threshold else "👤 Normal")

    # 3. GÖRSELLEŞTİRME (En hafif yöntem: Tablo ve Metin)
    st.subheader("📊 Analiz Sonuçları")
    st.write(f"Toplam Aktör Sayısı: {len(G.nodes())}")
    st.write(f"Toplam Bağlantı Sayısı: {len(G.edges())}")
    
    # Ağır grafikler yerine sadece en önemli 10 aktörü gösteren basit bir bar chart
    st.bar_chart(metrics_df.set_index('Aktör').head(10))
    
    st.divider()
    
    # Veri Tablosu
    st.dataframe(metrics_df)

    # 4. İNDİRME BUTONLARI (Alt alta)
    st.subheader("📂 Çıktıları Al")
    
    csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📄 Raporu İndir (CSV)", csv_data, f"hemithea_{current_user}.csv")
    
    # HTML Haritasını sadece indirme butonu olarak bırakıyoruz, ekrana çizmiyoruz!
    try:
        net_dl = Network(height="500px", width="100%")
        net_dl.from_nx(G)
        temp_path = os.path.join(tempfile.gettempdir(), f"map_{current_user}.html")
        net_dl.save_graph(temp_path)
        with open(temp_path, 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.download_button("🌐 İnteraktif Haritayı İndir (HTML)", html_string, "harita.html", "text/html")
    except:
        st.write("HTML Hazırlanamadı.")

elif isinstance(data_result, str):
    st.warning(f"Durum: {data_result}")
    if st.button("Tekrar Dene"): st.rerun()

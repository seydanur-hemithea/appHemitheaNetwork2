import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
from io import StringIO
import matplotlib.pyplot as plt 

# 1. Sayfa Ayarları
st.set_page_config(page_title="Hemithea Analiz", layout="wide")

# Mobil CSS
st.markdown("""
    <style>
    .main > div { padding: 0.5rem; }
    iframe { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# --- AYARLAR ---
BASE_RENDER_URL = "https://apphemitheanetwork.onrender.com/uploads"

# 2. Dinamik Veri Yükleme Fonksiyonu
@st.cache_data(ttl=2)
def load_dynamic_data(uname, token): # Token parametresi eklendi
    if not uname or not token:
        return None
    try:
        # FastAPI'ye giderken token'ı query parametresi olarak ekliyoruz
        target_url = f"{BASE_RENDER_URL}/{uname}/network_data.csv?token={token}"
        response = requests.get(target_url, timeout=5)
        
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        elif response.status_code == 401:
            st.error("🔑 Yetkisiz Erişim: Anahtarınız geçersiz veya süresi dolmuş.")
            return None
        return None
    except Exception as e:
        return None

# --- ANA AKIŞ ---
st.title("🌐 Hemithea Network Analytics")

# URL parametresinden user_name al
query_params = st.query_params
current_username = query_params.get("username")
current_token = query_params.get("token")

# --- DEDEKTİF PANELİ (SIDEBAR) ---

if current_username:
    st.sidebar.title("🔍 Sistem Denetimi")
    st.sidebar.info(f"Kullanıcı Adı: {current_username}")
    
    if st.sidebar.button("🔗 Bağlantıyı Manuel Test Et"):
        try:
            # Test ederken de token kullanmalıyız
            test_url = f"{BASE_RENDER_URL}/{current_username}/network_data.csv?token={current_token}"
            res = requests.get(test_url, timeout=5)
            if res.status_code == 200:
                st.sidebar.success(f"📡 Güvenli bağlantı kuruldu!")
            else:
                st.sidebar.error(f"📡 Erişim Reddedildi! (Kod: {res.status_code})")
        except Exception as e:
            st.sidebar.error(f"Bağlantı Hatası: {e}")


# Veriyi yüklemeyi dene
data = load_dynamic_data(current_username, current_token)


if data is not None:
    # Sütunları belirle
    cols = data.columns.tolist()
    src = cols[0]
    tgt = cols[1]

    st.success(f"✅ Analiz Hazır!")
    
    # Sekmeler
    tab1, tab2, tab3 = st.tabs(["🕸️ Ağ Haritası", "📈 Metrikler", "📄 Veri"])

    G = nx.from_pandas_edgelist(data, source=src, target=tgt)

    with tab1:
        st.subheader("Etkileşimli Ağ Haritası")
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        net.toggle_physics(True)
        html_content = net.generate_html()
        components.html(html_content, height=650)

        # --- PNG İNDİRME ÖZELLİĞİ ---
        st.divider()
        st.write("📸 Ağ Görüntüsünü Kaydet")
        
        # Statik bir kopya oluştur (İndirme için)
        fig, ax = plt.subplots(figsize=(10, 7))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', edge_color='gray', node_size=800, font_size=10)
        
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        
        st.download_button(
            label="Ağ Grafiğini PNG Olarak İndir",
            data=buf,
            file_name=f"hemithea_graph_{current_username}.png",
            mime="image/png"
        )
        plt.close(fig) # Belleği temizle

    with tab2:
        st.subheader("Ağ İstatistikleri")
        degree_cent = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        metrics_df = pd.DataFrame({
            'Aktör': list(degree_cent.keys()),
            'Bağlantı Skoru': list(degree_cent.values()),
            'Stratejik Rol': list(betweenness.values())
        }).sort_values(by='Bağlantı Skoru', ascending=False)

        st.dataframe(metrics_df, use_container_width=True)

        # --- CSV İNDİRME ÖZELLİĞİ ---
        st.divider()
        csv_data = metrics_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Metrikleri Tablo (CSV) Olarak İndir",
            data=csv_data,
            file_name=f"hemithea_metrics_{current_username}.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("Yüklenen Veri Tablosu")
        st.dataframe(data, use_container_width=True)

else:
    st.info("👋 Hoş geldiniz! Analiz edilecek veri bekleniyor...")
    if not current_username or not current_token:
        st.warning("⚠️ Güvenlik uyarısı: Uygulama üzerinden güvenli giriş yapmanız gerekiyor.")
    else:
        st.warning(f"{current_username} için dosya bulunamadı veya erişim yetkiniz yok.")

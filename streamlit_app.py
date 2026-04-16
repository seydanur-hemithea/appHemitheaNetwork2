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
            st.subheader("🤖 Yapay Zeka (KNN) ve Analitik Raporlama")
            
            # --- 1. KNN ANALİZ MOTORU ---
            if len(metrics_df) > 3:
                try:
                    # Özellikleri ölçeklendir (YZ Standartı)
                    X = metrics_df[['degree', 'betweenness']].values
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Stratejik Rol Belirleme (Betweenness ortalamasına göre etiketle)
                    y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
                    
                    # KNN Eğitimi
                    knn = KNeighborsClassifier(n_neighbors=min(3, len(metrics_df)-1))
                    knn.fit(X_scaled, y)
                    metrics_df['AI_Kategori'] = knn.predict(X_scaled)
                    metrics_df['Rol_Tanimi'] = metrics_df['AI_Kategori'].map({1: "Stratejik Köprü", 0: "Normal Aktör"})
                    
                    st.info("💡 KNN Modeli: Aktörler ağdaki stratejik konumlarına göre sınıflandırıldı.")
                except Exception as e:
                    st.error(f"YZ Analizi yapılamadı: {e}")
            
            # Analiz Tablosunu Göster
            st.dataframe(metrics_df, use_container_width=True)

            st.divider()
            
            # --- 2. ÇIKTI MERKEZİ (İNDİRME BUTONLARI) ---
            st.write("📂 **Analiz Çıktılarını İndir**")
            
            # İlk Satır: Veri ve Etkileşimli Dosya
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel Uyumlu CSV (utf-8-sig)
                csv_data = metrics_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📄 Veri Raporunu İndir (CSV)",
                    data=csv_data,
                    file_name=f"analiz_raporu_{current_user}.csv",
                    mime="text/csv",
                    key="dl_csv_final"
                )

            with col2:
                # Etkileşimli HTML Dosyası (Tıklanabilir Ağ)
                try:
                    net_dl = Network(height="600px", width="100%", bgcolor="#ffffff")
                    net_dl.from_nx(G)
                    net_dl.toggle_physics(True)
                    html_content = net_dl.generate_html()
                    
                    st.download_button(
                        label="🌐 Etkileşimli Ağı İndir (HTML)",
                        data=html_content,
                        file_name=f"interaktif_ag_{current_user}.html",
                        mime="text/html",
                        key="dl_html_final"
                    )
                except:
                    st.write("HTML Hazırlanıyor...")

            # İkinci Satır: Görsel (Resim) Raporlar
            st.write("📸 **Görsel (Resim) Kayıtları**")
            col3, col4 = st.columns(2)

            with col3:
                # Tabloyu Resim (PNG) Olarak İndir
                try:
                    plt.clf()
                    fig_tbl, ax_tbl = plt.subplots(figsize=(10, 6))
                    ax_tbl.axis('off')
                    # İlk 20 satırı resme dök (Okunabilirlik için)
                    plot_df = metrics_df.head(20)
                    the_table = ax_tbl.table(cellText=plot_df.values, colLabels=plot_df.columns, 
                                            loc='center', cellLoc='center')
                    the_table.auto_set_font_size(False)
                    the_table.set_fontsize(9)
                    the_table.scale(1.2, 1.2)
                    
                    buf_tbl = BytesIO()
                    plt.savefig(buf_tbl, format="png", dpi=150, bbox_inches='tight')
                    plt.close(fig_tbl)
                    
                    st.download_button(
                        label="🖼️ Tabloyu Resim Yap",
                        data=buf_tbl.getvalue(),
                        file_name=f"analiz_tablo_{current_user}.png",
                        mime="image/png",
                        key="dl_table_img"
                    )
                except:
                    st.write("Tablo resmi hazırlanıyor...")

            with col4:
                # Ağ Grafiğini Resim (PNG) Olarak İndir (Düğümler arası mesafeli)
                try:
                    plt.clf()
                    fig_gr, ax_gr = plt.subplots(figsize=(10, 8))
                    # k=0.5 düğümlerin birbirine girmesini önler
                    pos = nx.spring_layout(G, k=0.5) 
                    nx.draw(G, pos, with_labels=True, node_color='#3498db', 
                            node_size=400, font_size=7, edge_color='#ecf0f1')
                    
                    buf_gr = BytesIO()
                    plt.savefig(buf_gr, format="png", dpi=150, bbox_inches='tight')
                    plt.close(fig_gr)
                    
                    st.download_button(
                        label="📸 Ağı Resim Yap",
                        data=buf_gr.getvalue(),
                        file_name=f"ag_semasi_{current_user}.png",
                        mime="image/png",
                        key="dl_graph_img"
                    )
                except:
                    st.write("Grafik resmi hazırlanıyor...")

elif data_result == "NOT_FOUND":
        st.info(f"🔍 Hoş geldin {uname}! Henüz analiz edilecek bir verin yüklü değil.")
        
        # Kullanıcıyı yönlendiren bilgilendirme kutusu
        with st.expander("❓ Verimi Nasıl Yüklerim?", expanded=True):
            st.markdown("""
            1.  **Uygulama Ana Sayfasına Dön:** Hemithea mobil uygulamasından veri yükleme ekranına git.
            2.  **Dosyanı Seç:** `.csv` formatındaki ağ verilerini sisteme yükle.
            3.  **Analiz Et:** Yükleme tamamlandıktan sonra bu sayfa otomatik olarak güncellenecek (veya yukarıdaki 'Sistemi Yenile' butonuna basabilirsin).
            """)
        
        # Kullanıcının manuel tetiklemesi için buton
        if st.button("🔄 Veriyi Şimdi Kontrol Et"):
            st.rerun()

    elif data_result == "EMPTY":
        st.warning("⚠️ Dosya bulundu ancak içeriği boş!")
        st.write("Lütfen CSV dosyanızın başlıklarını ve verilerini kontrol edip tekrar yükleyin.")

    elif isinstance(data_result, str) and "ERROR" in data_result:
        st.error("📡 Sunucu ile bağlantı kurulamıyor.")
        st.info("Render sunucusu uyku modunda olabilir, lütfen birkaç saniye bekleyip sayfayı yenileyin.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Supermarket Sales Dashboard", layout="wide")

# --- DASHBOARD HEADER ---
st.title("📊 Supermarket Sales Analytics (KDD Implementation)")
st.caption("Dibuat oleh: Chairul Iman (23.230.0091)")
st.markdown("---")

# --- 1. LOAD DATA ---
@st.cache_data 
def load_data():
    try:   
        # Ganti dengan path file Anda jika perlu
        df = pd.read_csv('Supermarket Sales Cleaned.csv') 
    except:
        # Data Dummy Generator (Fallback jika file tidak ada)
        np.random.seed(42)
        n_rows = 1000
        data = {
            'Invoice ID': [f"{x}-{y}" for x, y in zip(np.random.randint(100, 999, n_rows), np.random.randint(10, 99, n_rows))],
            'Branch': np.random.choice(['A', 'B', 'C'], n_rows),
            'City': np.random.choice(['Yangon', 'Mandalay', 'Naypyitaw'], n_rows),
            'Customer type': np.random.choice(['Member', 'Normal'], n_rows),
            'Gender': np.random.choice(['Male', 'Female'], n_rows),
            'Product line': np.random.choice(['Health and beauty', 'Electronic accessories', 'Home and lifestyle', 'Sports and travel', 'Food and beverages', 'Fashion accessories'], n_rows),
            'Unit price': np.round(np.random.uniform(10, 100, n_rows), 2),
            'Quantity': np.random.randint(1, 11, n_rows),
            'Date': pd.date_range(start='1/1/2019', periods=n_rows).strftime('%m/%d/%Y'),
            'Time': [f"{x:02d}:{y:02d}" for x, y in zip(np.random.randint(10, 21, n_rows), np.random.randint(0, 59, n_rows))],
            'Payment': np.random.choice(['Cash', 'E-wallet', 'Credit card'], n_rows),
            'Rating': np.round(np.random.uniform(4, 10, n_rows), 1)
        }
        df = pd.DataFrame(data)
        df['Total'] = df['Unit price'] * df['Quantity'] * 1.05 
    
    # Preprocessing Wajib
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day_Name'] = df['Date'].dt.day_name()
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
    return df

# Load Data
df = load_data()

# --- SIDEBAR (FILTER) ---
st.sidebar.header("Filter Data")
selected_branch = st.sidebar.multiselect(
    "Pilih Cabang:",
    options=df["Branch"].unique(),
    default=df["Branch"].unique()
)

selected_gender = st.sidebar.multiselect(
    "Pilih Gender:",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

# Terapkan Filter
df_filtered = df.query("Branch == @selected_branch & Gender == @selected_gender")

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transaksi", f"{len(df_filtered)}")
col2.metric("Total Pendapatan", f"${df_filtered['Total'].sum():,.2f}")
col3.metric("Rata-rata Rating", f"{df_filtered['Rating'].mean():.1f}")
col4.metric("Cabang Terlaris", f"{df_filtered['Branch'].mode()[0] if not df_filtered.empty else '-'}")

st.markdown("---")

# --- PROSES CLUSTERING (K-MEANS) ---
if not df_filtered.empty:
    features = ['Total', 'Rating', 'Quantity']
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_filtered[features])
    
    k_clusters = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=5, value=3)
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    df_filtered['Cluster'] = kmeans.fit_predict(df_scaled)
    
    # Score
    score = silhouette_score(df_scaled, df_filtered['Cluster'])
else:
    st.error("Data kosong berdasarkan filter yang dipilih.")
    st.stop()

# --- TABS VISUALISASI ---
# KITA NAMBAH SATU TAB BARU: "Deep Dive Insights"
tab1, tab2, tab3, tab4 = st.tabs(["📈 Analisis Bisnis", "🔍 Deep Dive Insights", "🧩 Hasil Clustering", "📋 Data Mentah"])

# === TAB 1: ANALISIS BISNIS DASAR ===
with tab1:
    st.header("Analisis Performa Bisnis")
    
    # Row 1
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Tren Jam Sibuk")
        hourly_counts = df_filtered.groupby('Hour').size().reset_index(name='Counts')
        fig_line = plt.figure(figsize=(10, 5))
        sns.lineplot(data=hourly_counts, x='Hour', y='Counts', marker='o', color='red', linewidth=2.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Jam Operasional")
        plt.ylabel("Jumlah Transaksi")
        st.pyplot(fig_line)
        
    with col_b:
        st.subheader("Pendapatan per Lini Produk")
        product_sales = df_filtered.groupby('Product line')['Total'].sum().sort_values(ascending=False).reset_index()
        fig_bar = plt.figure(figsize=(10, 5))
        sns.barplot(data=product_sales, y='Product line', x='Total', palette='magma')
        plt.xlabel("Total Pendapatan ($)")
        st.pyplot(fig_bar)
    
    # Row 2
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.subheader("Metode Pembayaran")
        fig_pie, ax = plt.subplots()
        df_filtered['Payment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90, colors=sns.color_palette('pastel'))
        ax.set_ylabel('')
        st.pyplot(fig_pie)

    with col_d:
        st.subheader("Rating per Cabang")
        fig_box = plt.figure(figsize=(10, 5))
        sns.boxplot(data=df_filtered, x='Branch', y='Rating', palette='Set2')
        st.pyplot(fig_box)

# === TAB 2: DEEP DIVE INSIGHTS (BARU) ===
with tab2:
    st.header("Analisis & Insight Mendalam")

    # 1. Automated Text Insight (Kesimpulan Otomatis)
    st.subheader("💡 Key Insights (Dihasilkan Otomatis)")
    
    # Hitung data untuk insight
    top_day = df_filtered['Day_Name'].mode()[0]
    top_product = df_filtered.groupby('Product line')['Total'].sum().idxmax()
    avg_spend = df_filtered['Total'].mean()
    
    st.success(f"""
    Berdasarkan data yang difilter:
    - **Hari Teramai:** Pengunjung paling sering datang pada hari **{top_day}**.
    - **Produk Juara:** Kategori **{top_product}** memberikan pendapatan terbesar.
    - **Rata-rata Belanja:** Setiap pelanggan menghabiskan rata-rata **${avg_spend:.2f}** per transaksi.
    """)

    st.markdown("---")

    # 2. Grafik Hubungan Gender & Produk
    col_deep1, col_deep2 = st.columns(2)
    
    with col_deep1:
        st.subheader("Siapa Membeli Apa? (Gender vs Produk)")
        # Crosstab heatmap
        cross_tab = pd.crosstab(df_filtered['Product line'], df_filtered['Gender'])
        fig_heat = plt.figure(figsize=(10, 6))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.ylabel("Kategori Produk")
        st.pyplot(fig_heat)
        st.caption("Grafik ini menunjukkan kategori produk mana yang dominan dibeli oleh Pria atau Wanita.")

    with col_deep2:
        st.subheader("Matriks Korelasi (Hubungan Variabel)")
        # Korelasi antar angka
        numeric_df = df_filtered[['Total', 'Quantity', 'Rating', 'Unit price']]
        corr = numeric_df.corr()
        fig_corr = plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        st.pyplot(fig_corr)
        st.caption("Warna Merah = Hubungan Kuat. Contoh: Jika Total & Quantity merah, artinya makin banyak barang, harga makin mahal.")

    st.markdown("---")
    
    # 3. Time Series Analysis (Omzet Harian)
    st.subheader("📈 Tren Pendapatan Harian (Selama 3 Bulan)")
    daily_sales = df_filtered.groupby('Date')['Total'].sum().reset_index()
    
    fig_ts = plt.figure(figsize=(15, 5))
    sns.lineplot(data=daily_sales, x='Date', y='Total', color='green', linewidth=2)
    plt.title("Fluktuasi Pendapatan Harian")
    plt.xlabel("Tanggal")
    plt.ylabel("Total Pendapatan ($)")
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_ts)

# === TAB 3: HASIL CLUSTERING ===
with tab3:
    st.header(f"Segmentasi Pelanggan (K-Means = {k_clusters})")
    st.write(f"**Silhouette Score:** {score:.4f} (Semakin mendekati 1, cluster semakin rapi)")
    
    col_cluster1, col_cluster2 = st.columns([2, 1])
    
    with col_cluster1:
        # Scatter Plot Cluster
        fig_cluster = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_filtered, x='Total', y='Rating', hue='Cluster', palette='viridis', s=100, alpha=0.8)
        plt.title("Peta Persebaran Cluster (Total Belanja vs Rating)")
        plt.xlabel("Total Belanja ($)")
        plt.ylabel("Rating Customer")
        st.pyplot(fig_cluster)
        
    with col_cluster2:
        # Statistik Cluster
        st.write("### Profil Karakteristik Cluster")
        cluster_stats = df_filtered.groupby('Cluster')[['Total', 'Rating', 'Quantity']].mean()
        st.dataframe(cluster_stats.style.highlight_max(axis=0, color='lightgreen'))
        
        st.info("""
        **Panduan Analisis:**
        - **High Spenders:** Cluster dengan rata-rata 'Total' paling tinggi.
        - **At Risk:** Cluster dengan 'Rating' rendah.
        - **Loyal:** Cluster dengan 'Total' tinggi dan 'Rating' tinggi.
        """)
        
    # Boxplot Cluster (Tambahan Insight Clustering)
    st.subheader("Distribusi Rating per Cluster")
    fig_cluster_box = plt.figure(figsize=(12, 4))
    sns.boxplot(data=df_filtered, x='Cluster', y='Rating', palette='viridis')
    st.pyplot(fig_cluster_box)

# === TAB 4: DATA MENTAH ===
with tab4:
    st.header("Data Transaksi")
    st.dataframe(df_filtered)
    
    # Tombol Download
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data Filtered (CSV)",
        data=csv,
        file_name='supermarket_sales_filtered.csv',
        mime='text/csv',
    )

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

# --- DASHBOARD ---
st.title("Supermarket Sales Analytics (KDD Implementation)")
st.caption("Dibuat oleh: Chairul Iman (23.230.0091)")
st.markdown("---")

# --- 1. LOAD DATA ---
@st.cache_data 
def load_data():
    
    try:   
        df = pd.read_csv('Supermarket Sales Cleaned.csv') 
    except:
        # Data Dummy Generator
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
tab1, tab2, tab3 = st.tabs(["Analisis Bisnis", "Hasil Clustering", "Data Mentah"])

with tab1:
    st.header("Analisis Performa Bisnis")
    
    # Row 1
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Tren Jam Sibuk")
        hourly_counts = df_filtered.groupby('Hour').size().reset_index(name='Counts')
        fig_line = plt.figure(figsize=(10, 5))
        sns.lineplot(data=hourly_counts, x='Hour', y='Counts', marker='o', color='red')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_line)
        
    with col_b:
        st.subheader("Pendapatan per Lini Produk")
        fig_bar = plt.figure(figsize=(10, 5))
        sns.barplot(data=df_filtered, y='Product line', x='Total', estimator=sum, palette='magma')
        st.pyplot(fig_bar)
    
    # Row 2
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.subheader("Metode Pembayaran")
        fig_pie, ax = plt.subplots()
        df_filtered['Payment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
        ax.set_ylabel('')
        st.pyplot(fig_pie)

    with col_d:
        st.subheader("Rating per Cabang")
        fig_box = plt.figure(figsize=(10, 5))
        sns.boxplot(data=df_filtered, x='Branch', y='Rating', palette='Set2')
        st.pyplot(fig_box)

with tab2:
    st.header(f"Segmentasi Pelanggan (K-Means = {k_clusters})")
    st.write(f"**Silhouette Score:** {score:.4f} (Kualitas Cluster)")
    
    col_cluster1, col_cluster2 = st.columns([2, 1])
    
    with col_cluster1:
        # Scatter Plot Cluster
        fig_cluster = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_filtered, x='Total', y='Rating', hue='Cluster', palette='viridis', s=80)
        plt.title("Peta Persebaran Cluster (Total Belanja vs Rating)")
        st.pyplot(fig_cluster)
        
    with col_cluster2:
        # Statistik Cluster
        st.write("### Profil Rata-rata Cluster")
        cluster_stats = df_filtered.groupby('Cluster')[['Total', 'Rating', 'Quantity']].mean()
        st.dataframe(cluster_stats)
        
        st.info("""
        **Tips Membaca:**
        - Lihat nilai rata-rata 'Total' tertinggi -> Pelanggan Premium.
        - Lihat nilai 'Rating' terendah -> Pelanggan Tidak Puas.
        """)

with tab3:
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

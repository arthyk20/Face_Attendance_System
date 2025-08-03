import streamlit as st
import pandas as pd
from datetime import datetime

# === Load Data ===
CSV_PATH = "D:/FaceAttendence/attendance_logs.csv"

st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st.title("📊 Attendance Dashboard")

# Read CSV
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error("Attendance CSV file not found.")
    st.stop()

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar Filters
st.sidebar.header("🔍 Filters")

names = df['Name'].unique().tolist()
selected_name = st.sidebar.selectbox("Select Name", ["All"] + names)

dates = df['Date'].dt.date.unique().tolist()
selected_date = st.sidebar.selectbox("Select Date", ["All"] + sorted(dates, reverse=True))

# Filter logic
filtered_df = df.copy()

if selected_name != "All":
    filtered_df = filtered_df[filtered_df['Name'] == selected_name]

if selected_date != "All":
    filtered_df = filtered_df[filtered_df['Date'].dt.date == selected_date]

# Show total records
st.success(f"Showing {len(filtered_df)} records")

# Display Data
st.dataframe(filtered_df.sort_values(by="Date", ascending=False))

# Optional: Download button
st.download_button("📥 Download Filtered Data as CSV",
                   data=filtered_df.to_csv(index=False),
                   file_name="filtered_attendance.csv",
                   mime="text/csv")

st.markdown("---")
st.markdown("Face Attendance System - Attendance DataBase")

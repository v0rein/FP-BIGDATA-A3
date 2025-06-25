import streamlit as st
import pandas as pd
from kafka import KafkaConsumer
import json
from minio import Minio
from datetime import datetime
import io
import time

# Initialize MinIO client
minio_client = Minio(
    "localhost:9000",
    access_key="minio_access_key",
    secret_key="minio_secret_key",
    secure=False
)

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'tapive_playstore_dataset',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Initialize session state
if 'apps' not in st.session_state:
    st.session_state.apps = []
    st.session_state.last_save_time = time.time()

# Create bucket and initialize batch buffer
bucket_name = "apps"
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

# Batch buffer
buffer = []
last_save = time.time()


def save_to_minio(data_list):
    df = pd.DataFrame(data_list)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # Create a unique filename with timestamp
    filename = f"streamed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Save to MinIO
    minio_client.put_object(
        "apps",
        filename,
        io.BytesIO(csv_buffer.getvalue().encode()),
        len(csv_buffer.getvalue())
    )
    st.success(f"Saved batch to MinIO: {filename}")


# Streamlit app
st.title('Apps stream analysis')

# Create columns for metrics
col1, col2, col3 = st.columns(3)

# Initialize placeholder for the chart
chart_placeholder = st.empty()

# Initialize row for metrics
metrics_row = st.empty()

# Main loop with batch processing
while True:
    for msg in consumer:
        app = msg.value
        buffer.append(app)

        # Add app to session state for UI updates
        st.session_state.apps.append(app)

        now = time.time()
        app_title = app.get('title', 'Unknown App')
        print(f"[📥] Received app: {app_title}")

        # Save batch every 2 minutes
        if now - last_save >= 120:  # Save every 2 minutes
            if buffer:
                df = pd.DataFrame(buffer)
                filename = f"apps_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)

                minio_client.put_object(
                    bucket_name,
                    filename,
                    io.BytesIO(csv_buf.getvalue().encode()),
                    length=len(csv_buf.getvalue()),
                    content_type='application/csv'
                )

                print(f"[✅] Saved {len(buffer)} apps to MinIO as '{filename}'")
                st.success(
                    f"Saved {len(buffer)} apps to MinIO as '{filename}'")
                buffer.clear()
                last_save = now

        # Update metrics
        with metrics_row.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Apps Streamed", len(st.session_state.apps))
            with col2:
                avg_score = sum(float(app.get('score', 0)) for app in st.session_state.apps if app.get(
                    'score', 0) != 0) / max(len([app for app in st.session_state.apps if app.get('score', 0) != 0]), 1)
                st.metric("Average Score", f"{avg_score:.2f}")
            with col3:
                avg_ratings = sum(int(app.get('ratings', 0)) for app in st.session_state.apps if app.get(
                    'ratings', 0) != 0) / max(len([app for app in st.session_state.apps if app.get('ratings', 0) != 0]), 1)
                st.metric("Average Ratings Count", f"{avg_ratings:.0f}")

        # Update chart
        df = pd.DataFrame(st.session_state.apps)
        if not df.empty and 'score' in df.columns:
            chart_placeholder.line_chart(df['score'])

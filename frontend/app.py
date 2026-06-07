import streamlit as st # type: ignore
import pandas as pd # type: ignore
import plotly.graph_objects as go # type: ignore
from PIL import Image
import os
import time
import uuid
import io
import requests
import logging
from typing import Any
from datetime import datetime

if "backend_status" not in st.session_state:
    st.session_state.backend_status = {"healthy": False, "checked_at": None}
if "backend_url" not in st.session_state:
    st.session_state.backend_url = (
        os.getenv("BACKEND_URL")
        or st.secrets.get("BACKEND_URL")
        or st.secrets.get("backend", {}).get("url")
        or ""
    )

# initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("frontend")

BACKEND_URL = st.session_state.backend_url or "http://localhost:8000"


def get_backend_url() -> str:
    if st.session_state.backend_url:
        return st.session_state.backend_url

    candidates = [
        os.getenv("BACKEND_URL"),
        st.secrets.get("BACKEND_URL"),
        st.secrets.get("backend", {}).get("url"),
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://backend:8000",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if check_backend_health_once(candidate):
            st.session_state.backend_url = candidate
            logger.info("Resolved backend URL to %s", candidate)
            return candidate

    st.session_state.backend_url = BACKEND_URL
    return BACKEND_URL


def fetch_dataset_stats() -> dict[str, int]:
    success, response = make_api_request("GET", "/dataset-stats")
    if success and isinstance(response, dict):
        return {
            "total_classes": int(response.get("total_classes", 0)),
            "total_images": int(response.get("total_images", 0)),
        }
    logger.debug("Could not fetch backend dataset stats: %s", response)
    return {
        "total_classes": st.session_state.dataset_stats["classes"],
        "total_images": st.session_state.dataset_stats["images"],
    }


def check_backend_health_once(url: str, timeout: int = 2) -> bool:
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        if response.status_code == 200:
            # allow both {'status':'healthy'} and older formats
            try:
                payload = response.json()
                return payload.get("status") in ("healthy", "ok")
            except Exception:
                return True
        return False
    except requests.RequestException as e:
        logger.debug(f"Health check request failed: {e}")
        return False


def ensure_backend_available(timeout_sec: int = 5, poll_interval: float = 1.0) -> bool:
    """Poll backend /health until available or timeout.

    Blocks up to `timeout_sec` seconds, polling every `poll_interval` seconds.
    """
    start = time.time()
    backend_url = get_backend_url()
    logger.info("Checking backend availability at %s", backend_url)
    while time.time() - start < timeout_sec:
        if check_backend_health_once(backend_url):
            logger.info("Backend is healthy")
            st.session_state.backend_status.update(
                {
                    "healthy": True,
                    "checked_at": datetime.now().isoformat(),
                    "url": backend_url,
                }
            )
            return True
        logger.info("Backend not ready, sleeping %.1fs", poll_interval)
        time.sleep(poll_interval)
    logger.warning("Backend remained unavailable after %s seconds", timeout_sec)
    st.session_state.backend_status.update(
        {
            "healthy": False,
            "checked_at": datetime.now().isoformat(),
            "url": backend_url,
        }
    )
    return False


def wake_backend(timeout_sec: int = 10, poll_interval: float = 1.0) -> bool:
    st.session_state.backend_status.update(
        {
            "healthy": False,
            "checked_at": datetime.now().isoformat(),
            "url": get_backend_url(),
        }
    )
    return ensure_backend_available(timeout_sec=timeout_sec, poll_interval=poll_interval)


def make_api_request(
    method: str, endpoint: str, max_retries: int = 3, **kwargs
) -> tuple[bool, Any]:
    """Make an API request with retry logic. Returns (success, response_or_error)."""
    url = f"{get_backend_url().rstrip('/')}{endpoint}"

    for attempt in range(1, max_retries + 1):
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=10, **kwargs)
            elif method.upper() == "POST":
                response = requests.post(url, timeout=120, **kwargs)
            else:
                return False, {"error": "Invalid HTTP method"}

            if response.status_code == 200:
                try:
                    return True, response.json()
                except Exception:
                    return True, {"status_code": response.status_code}
            elif 400 <= response.status_code < 500:
                try:
                    return False, response.json()
                except Exception:
                    return False, {"error": response.text or f"HTTP {response.status_code}"}
            else:
                logger.warning("Server error %s on %s (attempt %d)", response.status_code, url, attempt)
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                try:
                    return False, response.json()
                except Exception:
                    return False, {"error": f"Server error {response.status_code}"}

        except requests.Timeout:
            logger.warning("Timeout on request to %s (attempt %d)", url, attempt)
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, {"error": "Request timeout. Please try again."}
        except requests.ConnectionError:
            logger.warning("Connection error to %s (attempt %d)", url, attempt)
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, {"error": "Connection failed. Backend may not be ready."}
        except Exception as e:
            logger.exception("Unexpected error during API request to %s: %s", url, e)
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, {"error": "An unexpected error occurred."}

    return False, {"error": "Request failed after retries"}


# ============================================================================
# Streamlit Configuration
# ============================================================================

st.set_page_config(
    page_title="AI Teachable Machine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body {
        background-color: #0b1220;
        color: #e5e7eb;
    }
    .css-1r6slb0 {
        background-color: #111827;
    }
    .css-1d391kg {
        background-color: #111827;
    }
    .st-bb {
        color: #e5e7eb;
    }
    .stButton>button {
        background-color: #2563eb;
        color: #ffffff;
    }
    .stTextInput>div>div>input,
    .stFileUploader>div>div,
    .stSelectbox>div>div>div>div {
        background-color: #1f2937;
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Session State Initialization
# ============================================================================

if "dataset_stats" not in st.session_state:
    st.session_state.dataset_stats = {"classes": 0, "images": 0}
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "status" not in st.session_state:
    st.session_state.status = "Ready"
if "model_status" not in st.session_state:
    st.session_state.model_status = "Not Trained"

# Get backend status by polling for up to 5 seconds
backend_url = get_backend_url()
backend_health_ok = ensure_backend_available(timeout_sec=5, poll_interval=1.0)
backend_status_msg = "Online" if backend_health_ok else "Offline"
backend_last_checked = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
backend_unavailable_message = (
    "We're having trouble connecting to our services.\n"
    "Don't worry — this is usually temporary. Try the following:\n\n"
    "1. Check your internet connection\n"
    "2. Click 'Wake Backend' above to bring the service back online\n\n"
    "If the problem persists, try:\n"
    "• Refreshing the page\n"
    "• Clearing your browser cache\n"
    "• Using a different browser\n\n"
    "We're working to restore service. Thank you for your patience!"
)
backend_stats = fetch_dataset_stats() if backend_health_ok else {
    "total_classes": st.session_state.dataset_stats["classes"],
    "total_images": st.session_state.dataset_stats["images"],
}

# ============================================================================
# Sidebar Navigation
# ============================================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712340.png", width=75)
    st.title("AI Teachable Machine")
    section = st.radio(
        "Navigation",
        ["Dataset", "Training", "Prediction", "About"],
        index=0,
    )
    st.markdown("---")

    status_label = "Ready" if backend_health_ok else "Offline"
    status_icon = "✅" if backend_health_ok else "❌"
    st.markdown(f"{status_icon} **{status_label}**")
    st.caption("Backend connection is managed automatically behind the scenes.")

    if not backend_health_ok:
        st.error(backend_unavailable_message)
        button_pressed = st.button("Wake Backend")
        if button_pressed:
            with st.spinner("Waking backend service..."):
                if wake_backend(timeout_sec=10, poll_interval=1.0):
                    st.success("Backend is awake and ready.")
                    try:
                        st.rerun()
                    except Exception:
                        if hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()
                else:
                    st.error("Backend is still unavailable. Please try again or check your backend service.")

    st.markdown("---")
    st.write("Build your custom AI visual classifier with a clean training workflow.")

# ============================================================================
# Main Dashboard Header
# ============================================================================

st.markdown("# AI Teachable Machine Dashboard")
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown(
        "Build a teachable image classifier with a modern Streamlit dashboard and a FastAPI backend."
    )
    st.caption(
        "Use the sidebar to navigate between dataset creation, model training, and prediction."
    )
with col2:
    info_cols = st.columns(3)
    info_cols[0].metric(
        "System Status",
        "Ready" if backend_health_ok else "Offline",
        "Auto-managed",
    )
    info_cols[1].metric("Dataset Classes", backend_stats.get("total_classes", 0))
    info_cols[2].metric("Dataset Images", backend_stats.get("total_images", 0))

st.markdown("---")

status_cols = st.columns(4)
status_cols[0].metric("Model Status", st.session_state.model_status)
status_cols[1].metric("Training Status", st.session_state.status)
status_cols[2].metric("Data Classes", backend_stats.get("total_classes", 0))
status_cols[3].metric("Data Images", backend_stats.get("total_images", 0))

st.markdown("---")
with st.container():
    st.markdown("### Dashboard Highlights")
    highlight_cols = st.columns(3)
    highlight_cols[0].metric("Fast Iteration", "Upload → Train → Predict", "One-click workflow")
    highlight_cols[1].metric("Data Quality", "Preview and validate datasets", "CSV, Excel, images")
    highlight_cols[2].metric("Visual Output", "Live prediction charts", "Confidence and probabilities")
    st.write(
        "This interface lets you manage dataset uploads, run training, and validate predictions in a polished Streamlit experience."
    )

# ============================================================================
# Dataset Section
# ============================================================================

if section == "Dataset":
    st.subheader("Dataset Creation")
    st.write("Add labeled images, preview upload samples, and keep your dataset ready for training.")

    if not backend_health_ok:
        st.error(backend_unavailable_message)
        if st.button("Retry connection"):
            try:
                st.rerun()
            except Exception:
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
        st.stop()

    dataset_left, dataset_right = st.columns([2, 1])
    with dataset_left:
        with st.form("dataset_form"):
            class_name = st.text_input("Class Name", placeholder="Enter class label")
            uploaded_files = st.file_uploader(
                "Upload images for this class",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )
            camera_image = st.camera_input("Capture an image")
            upload_button = st.form_submit_button("Upload Samples")

        if upload_button:
            if not class_name.strip():
                st.error("Please provide a class name before uploading samples.")
            elif not uploaded_files and camera_image is None:
                st.error("Upload or capture at least one image.")
            else:
                files_payload = []
                for uploaded_file in uploaded_files:
                    files_payload.append(
                        (
                            "files",
                            (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            ),
                        )
                    )
                if camera_image is not None:
                    img_bytes = camera_image.getvalue()
                    files_payload.append(
                        (
                            "files",
                            (f"camera_{uuid.uuid4().hex}.png", img_bytes, "image/png"),
                        )
                    )

                with st.spinner("Uploading samples..."):
                    success, response = make_api_request(
                        "POST",
                        "/upload-sample",
                        data={"class_name": class_name.strip()},
                        files=files_payload,
                    )

                if success:
                    st.success("Samples uploaded successfully!")
                    st.session_state.dataset_stats["classes"] = response.get(
                        "total_classes", st.session_state.dataset_stats["classes"]
                    )
                    st.session_state.dataset_stats["images"] = response.get(
                        "total_images", st.session_state.dataset_stats["images"]
                    )
                    backend_stats["total_classes"] = st.session_state.dataset_stats["classes"]
                    backend_stats["total_images"] = st.session_state.dataset_stats["images"]
                else:
                    st.error(response.get("detail") if isinstance(response, dict) else "Upload failed. Please try again.")

    with dataset_right:
        st.markdown("### Dataset Snapshot")
        st.write("Current dataset statistics retrieved from the backend.")
        st.metric("Classes", backend_stats.get("total_classes", 0))
        st.metric("Images", backend_stats.get("total_images", 0))
        st.info(
            "Use the sample upload form to add new labeled images. Then go to the Training tab to build your model."
        )

    st.markdown("---")
    st.subheader("📊 Advanced Dataset Tools")
    st.write("Preview CSV/Excel datasets or inspect images before training.")

    upload_tab1, upload_tab2, upload_tab3 = st.tabs(
        ["📁 CSV Dataset", "📑 Excel Dataset", "🖼️ Image Preview"]
    )

    # CSV Upload Tab
    with upload_tab1:
        st.markdown("#### Upload CSV Dataset")
        csv_file = st.file_uploader(
            "Choose a CSV file", type=["csv"], key="csv_uploader"
        )

        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
                st.success("✅ CSV loaded successfully!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric(
                        "Memory",
                        f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                    )

                st.markdown("##### Column Information")
                col_info = pd.DataFrame(
                    {
                        "Column": df.columns,
                        "Type": df.dtypes.values,
                        "Non-Null": df.count().values,
                        "Missing": df.isnull().sum().values,
                    }
                )
                st.dataframe(col_info, use_container_width=True, hide_index=True)
                st.markdown("##### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown("##### Dataset Summary")
                st.dataframe(df.describe(), use_container_width=True)
                csv_download = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_download,
                    file_name=csv_file.name,
                    mime="text/csv",
                )
            except pd.errors.EmptyDataError:
                st.error("CSV file is empty. Please upload a valid file.")
            except pd.errors.ParserError:
                st.error("Could not parse CSV file. Please check the format.")
            except Exception:
                logger.exception("CSV upload failed")
                st.error("An error occurred processing the file.")
        else:
            st.info("👆 Upload a CSV file to preview and analyze it.")

    # Excel Upload Tab
    with upload_tab2:
        st.markdown("#### Upload Excel Dataset")
        excel_file = st.file_uploader(
            "Choose an Excel file (.xlsx)",
            type=["xlsx"],
            key="excel_uploader",
        )

        if excel_file is not None:
            try:
                df = pd.read_excel(excel_file, sheet_name=0)
                st.success("✅ Excel loaded successfully!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric(
                        "Memory",
                        f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                    )

                st.markdown("##### Column Information")
                col_info = pd.DataFrame(
                    {
                        "Column": df.columns,
                        "Type": df.dtypes.values,
                        "Non-Null": df.count().values,
                        "Missing": df.isnull().sum().values,
                    }
                )
                st.dataframe(col_info, use_container_width=True, hide_index=True)
                st.markdown("##### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown("##### Dataset Summary")
                st.dataframe(df.describe(), use_container_width=True)
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False, sheet_name="Data")
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer,
                    file_name=f"processed_{excel_file.name}",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception:
                logger.exception("Excel upload failed")
                st.error("An error occurred processing the file.")
        else:
            st.info("👆 Upload an Excel file (.xlsx) to preview and analyze it.")

    # Image Upload Tab
    with upload_tab3:
        st.markdown("#### Upload Images")
        image_files = st.file_uploader(
            "Choose image files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="image_uploader",
        )

        if image_files:
            st.success(f"✅ {len(image_files)} image(s) loaded!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Images", len(image_files))
            with col2:
                total_size = sum(len(img.getvalue()) for img in image_files)
                st.metric("Total Size", f"{total_size / (1024 * 1024):.2f} MB")

            st.markdown("##### Image Preview Gallery")
            cols_per_row = 3
            cols = st.columns(cols_per_row)
            for idx, image_file in enumerate(image_files):
                try:
                    image = Image.open(image_file)
                    col = cols[idx % cols_per_row]
                    with col:
                        st.image(image, caption=image_file.name, use_column_width=True)
                        st.caption(
                            f"Size: {len(image_file.getvalue()) / 1024:.2f} KB"
                        )
                except Exception:
                    logger.exception(f"Error loading image {image_file.name}")
                    st.warning(f"Could not load: {image_file.name}")
        else:
            st.info("👆 Upload image files (.jpg, .jpeg, .png) to preview them.")

# ============================================================================
# Training Section
# ============================================================================

elif section == "Training":
    st.subheader("Train Model")
    st.write(
        "Train a classifier using MobileNetV3 feature extraction and a logistic regression head."
    )

    if not backend_health_ok:
        st.error(backend_unavailable_message)
        if st.button("Retry connection"):
            try:
                st.rerun()
            except Exception:
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
        st.stop()
    else:
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                progress = st.progress(0)
                st.session_state.status = "Training"
                progress.progress(10)

                success, response = make_api_request("POST", "/train")

                if success:
                    progress.progress(100)
                    st.success("Training completed successfully!")
                    st.session_state.dataset_stats["classes"] = response.get(
                        "total_classes", st.session_state.dataset_stats["classes"]
                    )
                    st.session_state.dataset_stats["images"] = response.get(
                        "total_images", st.session_state.dataset_stats["images"]
                    )
                    st.session_state.status = "Model trained"
                    st.session_state.model_status = "Trained"
                else:
                    st.session_state.status = "Training failed"
                    st.error("Training failed. Please try again.")

    st.markdown("### Training Summary")
    st.write("After training, use the dataset statistics to understand coverage.")
    summary_cols = st.columns(2)
    summary_cols[0].metric("Classes", st.session_state.dataset_stats["classes"])
    summary_cols[1].metric("Images", st.session_state.dataset_stats["images"])

# ============================================================================
# Prediction Section
# ============================================================================

elif section == "Prediction":
    st.subheader("Image Prediction")
    st.write("Upload a test image to receive a predicted class and confidence.")

    if not backend_health_ok:
        st.error(backend_unavailable_message)
        if st.button("Retry connection"):
            try:
                st.rerun()
            except Exception:
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
        st.stop()
    else:
        test_image = st.file_uploader(
            "Upload a test image", type=["png", "jpg", "jpeg"]
        )

        if st.button("Run Prediction"):
            if test_image is None:
                st.error("Please upload a test image to continue.")
            else:
                with st.spinner("Running prediction..."):
                    progress = st.progress(0)
                    progress.progress(20)

                    success, response = make_api_request(
                        "POST",
                        "/predict",
                        files={
                            "image": (
                                test_image.name,
                                test_image.getvalue(),
                                test_image.type,
                            )
                        },
                    )

                    progress.progress(80)

                    if success:
                        progress.progress(100)
                        label = response.get("predicted_class")
                        confidence = response.get("confidence", 0.0) * 100
                        st.session_state.last_prediction = {
                            "label": label,
                            "confidence": confidence,
                            "probabilities": response.get("probabilities", {}),
                        }
                        st.success(f"Prediction: {label}")
                        st.write(f"Confidence: {confidence:.2f}%")
                    else:
                        st.error("Prediction failed. Please try again.")

        if st.session_state.last_prediction:
            st.markdown("### Prediction Output")
            st.write(
                f"**{st.session_state.last_prediction['label']}** — {st.session_state.last_prediction['confidence']:.2f}%"
            )
            chart_data = {
                key: value * 100
                for key, value in st.session_state.last_prediction[
                    "probabilities"
                ].items()
            }
            if chart_data:
                fig = go.Figure(
                    go.Bar(
                        x=list(chart_data.keys()),
                        y=list(chart_data.values()),
                        marker_color="#38bdf8",
                    )
                )
                fig.update_layout(
                    title="Prediction Probabilities",
                    plot_bgcolor="#0b1220",
                    paper_bgcolor="#0b1220",
                    font=dict(color="#e5e7eb"),
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# About Section
# ============================================================================

else:
    st.subheader("About")
    st.write(
        "This interface uses Streamlit to interact with a FastAPI backend. Upload training samples, train a model, and make predictions from a modern dashboard."
    )
    st.markdown(
        "- Backend: FastAPI, PyTorch, MobileNetV3 feature extraction, scikit-learn logistic regression."
    )
    st.markdown(
        "- Frontend: Streamlit, Plotly, Requests, responsive dark-theme layout."
    )
    st.markdown(
        "- Model artifacts are persisted in backend/saved_models and training samples are saved in backend/dataset."
    )

import streamlit as st
import subprocess
import os
import shutil

st.set_page_config(page_title="Number Plate Detection", layout="centered")
st.title("🔍 Number Plate Detection using YOLOv8 + OpenCV")

# === VIDEO INPUT SECTION ===
st.markdown("### 🎥 Choose a video for detection")

import tempfile

uploaded_video = st.file_uploader("Upload a video file (MP4 format)", type=["mp4"])

if uploaded_video is not None:
    # Save to a temp file first
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(uploaded_video.read())
    temp_file.flush()
    temp_file.close()

    # Rename to sample.mp4 (overwrites if exists)   
    shutil.move(temp_file.name, "sample.mp4")

    st.success("✅ Video uploaded and saved as `sample.mp4`")

    # Optional: Re-encode to browser-compatible format
    subprocess.run([
        "ffmpeg", "-y", "-i", "sample.mp4",
        "-vcodec", "libx264", "-acodec", "aac",
        "sample_clean.mp4"
    ])

    os.replace("sample_clean.mp4", "sample.mp4")  # Use re-encoded file
    st.video("sample.mp4")
elif not os.path.exists("sample.mp4"):
    st.warning("⚠️ No video uploaded yet. Please upload an MP4 file.")

st.markdown("---")

# === PIPELINE BUTTONS ===
st.markdown("### 🧪 Run Pipeline Stages")

# -- Helper functions --
def run_script(name, file):
    with st.spinner(f"Running {name}..."):
        result = subprocess.run(["python", file], capture_output=True, text=True)
        st.code(result.stdout + "\n" + result.stderr)
        st.success(f"{name} completed!")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚗 Run Detection"):
        if os.path.exists("sample.mp4"):
            run_script("Detection", "main.py")
        else:
            st.error("Please upload a video first.")

with col2:
    if st.button("📈 Run Interpolation"):
        if os.path.exists("test.csv"):
            run_script("Interpolation", "add_missing_data.py")
        else:
            st.error("Please run Detection first to generate `test.csv`.")

with col3:
    if st.button("🎞️ Run Visualization"):
        if os.path.exists("test_interpolated.csv"):
            run_script("Visualization", "visualize.py")
        else:
            st.error("Please run Interpolation first to generate `test_interpolated.csv`.")

st.markdown("---")

# === OUTPUT VIDEO SECTION ===
st.markdown("### ✅ Output Video")
if os.path.exists("out.mp4"):
    st.video("out.mp4")

    # Optional: Download button if streaming fails
    with open("out.mp4", "rb") as f:
        st.download_button("⬇️ Download Output Video", f, file_name="out.mp4")
else:
    st.info("Final output video (`out.mp4`) will appear here after visualization.")

subprocess.run([
    "ffmpeg", "-y", "-i", "out.mp4",
    "-vcodec", "libx264", "-acodec", "aac",
    "-strict", "-2", "out_clean.mp4"
])
os.replace("out_clean.mp4", "out.mp4")
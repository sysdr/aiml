"""
Interactive dashboard to visualize the Day 32 NumPy vectorization lesson.
Run with: streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from lesson_code import (
    AdvancedIndexing,
    ImageBatchProcessor,
    PerformanceBenchmark,
    WeightInitializer,
)


st.set_page_config(
    page_title="Day 32 · NumPy Vectorization Dashboard",
    layout="wide",
    page_icon="📈",
)

st.title("Day 32 · NumPy Array Manipulation and Vectorization")
st.caption("Explore the lesson outputs interactively.")


def build_bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    fig = px.bar(df, x=x, y=y, text=y, title=title)
    fig.update_layout(yaxis_title="", xaxis_title="", height=320, margin=dict(t=60, l=10, r=10, b=10))
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    return fig


@st.cache_data(show_spinner=False)
def generate_image_batch(batch_size: int, height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (batch_size, height, width, 3), dtype=np.uint8)


@st.cache_data(show_spinner=False)
def run_performance(size: int) -> dict:
    return PerformanceBenchmark.run_benchmark(size=size)


# SECTION 1: Image Batch Processing ------------------------------------------------
st.header("1 · Image Batch Processing")

col_params, col_stats = st.columns([1, 2])

with col_params:
    batch_size = st.slider("Batch size", min_value=8, max_value=64, value=32, step=8)
    target_h = st.number_input("Target height", min_value=64, max_value=512, value=224, step=32)
    target_w = st.number_input("Target width", min_value=64, max_value=512, value=224, step=32)
    patch_size = st.selectbox("Patch size", options=[8, 16, 32], index=1)

images = generate_image_batch(batch_size, target_h, target_w)
processor = ImageBatchProcessor(target_size=(target_h, target_w))
processed = processor.prepare_batch(images)
patches = processor.extract_patches(images[0], patch_size=patch_size)

with col_stats:
    st.write("### Batch Stats")
    st.metric("Input shape", str(images.shape))
    st.metric("Output shape", str(processed.shape))
    st.metric("Output dtype", str(processed.dtype))
    st.metric(
        "Output range",
        f"[{processed.min():.2f}, {processed.max():.2f}]",
    )

st.write("### Patch Overview")
st.write(f"{patches.shape[0]} patches of size {patch_size}×{patch_size}")
patch_df = pd.DataFrame(
    {"Metric": ["Channels", "Patches"], "Value": [3, patches.shape[0]]}
)
st.plotly_chart(build_bar_chart(patch_df, "Metric", "Value", "Patch metrics"), use_container_width=True)


# SECTION 2: Weight Initialization -------------------------------------------------
st.header("2 · Neural Network Weight Initialization")

default_layers = "784,256,128,10"
layer_text = st.text_input(
    "Layer sizes (comma-separated)",
    value=default_layers,
    help="Example: 1024,512,256,10",
)

try:
    layer_sizes = [int(x.strip()) for x in layer_text.split(",") if x.strip()]
    if len(layer_sizes) < 2:
        raise ValueError
except ValueError:
    st.error("Enter at least two valid integers.")
    layer_sizes = [784, 256, 128, 10]

weights = WeightInitializer.initialize_network(layer_sizes)

cols = st.columns(len(weights))
for idx, (col, w) in enumerate(zip(cols, weights), start=1):
    with col:
        col.metric(f"Layer {idx}", str(w.shape))
        col.metric("mean", f"{w.mean():.4f}")
        col.metric("std", f"{w.std():.4f}")


# SECTION 3: Vectorization Performance --------------------------------------------
st.header("3 · Vectorization Performance Benchmark")

bench_size = st.slider("Matrix size", min_value=200, max_value=600, value=400, step=50)
results = run_performance(bench_size)

norm = results["normalize"]
matmul = results["matmul"]

bench_cols = st.columns(2)
norm_df = pd.DataFrame(
    {"Implementation": ["Loop", "Vectorized"], "Milliseconds": [norm["loop_ms"], norm["vectorized_ms"]]}
)
matmul_df = pd.DataFrame(
    {
        "Implementation": ["Loop", "Vectorized"],
        "Milliseconds": [matmul["loop_ms"], matmul["vectorized_ms"]],
    }
)

with bench_cols[0]:
    st.subheader("Normalization (ms)")
    st.plotly_chart(
        build_bar_chart(norm_df, "Implementation", "Milliseconds", "Normalization timing"),
        use_container_width=True,
    )
    st.metric("Speedup", f"{norm['speedup']:.1f}x")

with bench_cols[1]:
    st.subheader("Matrix Multiply (ms)")
    st.plotly_chart(
        build_bar_chart(matmul_df, "Implementation", "Milliseconds", "Matmul timing"),
        use_container_width=True,
    )
    st.metric("Speedup", f"{matmul['speedup']:.1f}x")


# SECTION 4: Advanced Indexing -----------------------------------------------------
st.header("4 · Advanced Indexing Playground")

score_count = st.slider("Number of scores", min_value=100, max_value=5000, value=1000, step=100)
top_k = st.slider("Top-k", min_value=1, max_value=20, value=5)
threshold = st.slider("Threshold", min_value=0.5, max_value=0.99, value=0.9, step=0.01)

rng = np.random.default_rng(123)
scores = rng.random(score_count)

top_indices = AdvancedIndexing.top_k_indices(scores, k=top_k)
filtered_values, filtered_idx = AdvancedIndexing.filter_by_threshold(scores, threshold)

st.write("### Top-k Selection")
st.dataframe(
    {
        "index": top_indices,
        "score": scores[top_indices],
    }
)

st.write("### Threshold Filtering")
st.metric("Detections above threshold", len(filtered_values))

scatter_target = np.zeros(10)
scatter_indices = np.array([1, 3, 3, 5, 5, 5])
scatter_values = np.ones_like(scatter_indices, dtype=float)
scatter_result = AdvancedIndexing.scatter_add(scatter_target, scatter_indices, scatter_values)

st.write("### Scatter-Add Result")
scatter_df = pd.DataFrame({"Index": np.arange(len(scatter_result)), "Value": scatter_result})
st.plotly_chart(build_bar_chart(scatter_df, "Index", "Value", "Scatter-add accumulation"), use_container_width=True)


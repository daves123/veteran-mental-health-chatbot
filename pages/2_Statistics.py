# pages/2_statistics.py

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path for RAG system import
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag_system import VeteranHealthRAG

st.set_page_config(page_title="System Statistics", page_icon="📈", layout="wide")

# Header
st.markdown(
    """
    <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1f4e78 0%, #2d6a9f 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'> System Statistics</h1>
        <p style='color: #e0e0e0; margin: 0.5rem 0 0 0;'>RAG System Performance & Data Insights</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("""
    Responses are generated using Retrieval-Augmented Generation (RAG),
    which combines semantic search over a clinical evidence corpus with natural language generation.
""")


# Load RAG system
@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system"""
    try:
        rag = VeteranHealthRAG(
            faiss_index_path="./models/faiss_index",
            chunks_path="./data/veteran_rag_corpus.csv",
        )

        try:
            rag.load_index()
            return rag
        except FileNotFoundError:
            st.warning(
                "⚠️ RAG index not found. Please run the main app first to build the index."
            )
            return None

    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None


rag = load_rag_system()

if rag is None:
    st.stop()

# Colorblind-friendly colors
COLORS = {
    "male": "#0173B2",  # Blue
    "female": "#DE8F05",  # Orange
    "neutral": "#029E73",  # Teal
    "primary": "#0173B2",
}

# Main metrics
st.header("Overview Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
            <h3 style='color: #1f4e78; margin: 0; font-size: 2rem;'>{len(rag.chunks_df):,}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>Knowledge Chunks</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
            <h3 style='color: #1f4e78; margin: 0; font-size: 2rem;'>{rag.faiss_index.ntotal:,}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>Indexed Vectors</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    sources_count = len(rag.chunks_df["source"].unique())
    st.markdown(
        f"""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
            <h3 style='color: #1f4e78; margin: 0; font-size: 2rem;'>{sources_count}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>Data Sources</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    embedding_dim = rag.faiss_index.d
    st.markdown(
        f"""
        <div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
            <h3 style='color: #1f4e78; margin: 0; font-size: 2rem;'>{embedding_dim}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>Embedding Dimensions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Gender Breakdown Analysis
st.header("⚧ Gender Breakdown Analysis")

st.info("""
**Why Gender Breakdown Matters:**
- Female veterans experience unique mental health challenges
- Military sexual trauma affects genders differently  
- Treatment responses may vary by gender
- Proper representation ensures inclusive responses
""")

# Extract gender information from metadata
gender_data = []
for idx, row in rag.chunks_df.iterrows():
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        gender = metadata.get("gender", "neutral")
    else:
        # Try to infer from source name
        source = str(row.get("source", "")).lower()
        if "female" in source:
            gender = "female"
        elif "male" in source and "female" not in source:
            gender = "male"
        else:
            gender = "neutral"
    gender_data.append(gender)

rag.chunks_df["gender_tag"] = gender_data

# Gender distribution
gender_counts = rag.chunks_df["gender_tag"].value_counts()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Gender Distribution of Knowledge Chunks")

    # Create pie chart with colorblind-friendly colors
    colors_list = [COLORS.get(g, COLORS["neutral"]) for g in gender_counts.index]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[f"{g.capitalize()} Veteran Data" for g in gender_counts.index],
                values=gender_counts.values,
                marker=dict(colors=colors_list),
                textinfo="label+percent",
                textfont_size=12,
                hole=0.3,
            )
        ]
    )

    fig.update_layout(title="Data Distribution by Gender", showlegend=True, height=400)

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Gender Statistics")

    total_chunks = len(rag.chunks_df)

    for gender in ["male", "female", "neutral"]:
        count = gender_counts.get(gender, 0)
        pct = (count / total_chunks * 100) if total_chunks > 0 else 0

        label_map = {
            "male": "👨‍✈️ Male Veteran Data",
            "female": "👩‍✈️ Female Veteran Data",
            "neutral": "🔷 Gender-Neutral Data",
        }

        st.metric(
            label_map.get(gender, gender.capitalize()),
            f"{count:,} chunks",
            f"{pct:.1f}% of total",
        )

    st.markdown("---")

    st.markdown("""
    **Data Sources by Gender:**
    - Female veteran-specific BRFSS data
    - Male veteran-specific BRFSS data
    - General PTSD research (all genders)
    - Treatment protocols (applicable to all)
    """)

st.markdown("---")

# Source Distribution
st.header("📚 Data Source Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Distribution by Source")

    source_counts = rag.chunks_df["source"].value_counts()

    # Show top sources in pie chart, group smaller ones
    top_n = 10
    top_sources = source_counts.head(top_n)
    other_count = source_counts[top_n:].sum()

    if other_count > 0:
        chart_data = pd.concat([top_sources, pd.Series({"Other sources": other_count})])
    else:
        chart_data = top_sources

    fig = px.pie(
        values=chart_data.values,
        names=chart_data.index,
        title=f"Top {top_n} Sources + Others",
        color_discrete_sequence=px.colors.sequential.Teal,
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
        margin=dict(l=20, r=250, t=60, b=20),
        height=500,
    )

    fig.update_traces(textposition="inside", textinfo="percent", textfont_size=11)

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top 10 Sources")

    top_10 = source_counts.head(10).reset_index()
    top_10.columns = ["Source", "Chunks"]
    top_10["Percentage"] = (top_10["Chunks"] / len(rag.chunks_df) * 100).round(1)

    st.dataframe(
        top_10,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source": st.column_config.TextColumn("Source", width="medium"),
            "Chunks": st.column_config.NumberColumn("Chunks", format="%d"),
            "Percentage": st.column_config.NumberColumn("Percentage", format="%.1f%%"),
        },
    )

st.markdown("---")

# Gender by Source Cross-Analysis
st.header("Gender Distribution Across Top Sources")

top_sources_list = source_counts.head(10).index.tolist()
gender_source_data = []

for source in top_sources_list:
    source_df = rag.chunks_df[rag.chunks_df["source"] == source]
    gender_dist = source_df["gender_tag"].value_counts()

    for gender in ["male", "female", "neutral"]:
        count = gender_dist.get(gender, 0)
        gender_source_data.append(
            {
                "Source": source[:30] + "..." if len(source) > 30 else source,
                "Gender": gender.capitalize(),
                "Count": count,
            }
        )

gender_source_df = pd.DataFrame(gender_source_data)

if not gender_source_df.empty:
    fig = px.bar(
        gender_source_df,
        x="Source",
        y="Count",
        color="Gender",
        title="Gender Distribution Across Top Data Sources",
        color_discrete_map={
            "Male": COLORS["male"],
            "Female": COLORS["female"],
            "Neutral": COLORS["neutral"],
        },
        barmode="stack",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        xaxis_title="Data Source",
        yaxis_title="Number of Chunks",
        legend_title="Gender Category",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:** This visualization shows how our data sources contribute to gender representation. 
    Sources explicitly labeled as 'Female' or 'Male' veteran data provide gender-specific insights, 
    while general sources (like PTSD research and treatment protocols) are applicable to all veterans.
    """)

st.markdown("---")

# All sources table
st.header("Complete Source Inventory")

with st.expander("View All Data Sources", expanded=False):
    all_sources = source_counts.reset_index()
    all_sources.columns = ["Source", "Number of Chunks"]
    all_sources["Percentage"] = (
        all_sources["Number of Chunks"] / len(rag.chunks_df) * 100
    ).round(2)

    st.dataframe(all_sources, use_container_width=True, hide_index=True, height=400)

    # Download button
    csv = all_sources.to_csv(index=False)
    st.download_button(
        label="Download Source Inventory (CSV)",
        data=csv,
        file_name="rag_source_inventory.csv",
        mime="text/csv",
    )

st.markdown("---")

# Chunk length analysis
st.header("Chunk Length Analysis")

col1, col2 = st.columns(2)

with col1:
    # Calculate chunk lengths
    rag.chunks_df["chunk_length"] = rag.chunks_df["text"].str.len()

    fig = px.histogram(
        rag.chunks_df,
        x="chunk_length",
        nbins=50,
        title="Distribution of Chunk Lengths (Characters)",
        labels={"chunk_length": "Characters", "count": "Number of Chunks"},
        color_discrete_sequence=[COLORS["primary"]],
    )

    fig.update_layout(showlegend=False, height=400)

    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Summary statistics
    st.subheader("Chunk Length Statistics")

    stats = rag.chunks_df["chunk_length"].describe()

    st.metric("Average Length", f"{stats['mean']:.0f} chars")
    st.metric("Median Length", f"{stats['50%']:.0f} chars")
    st.metric("Shortest Chunk", f"{stats['min']:.0f} chars")
    st.metric("Longest Chunk", f"{stats['max']:.0f} chars")
    st.metric("Standard Deviation", f"{stats['std']:.0f} chars")

    st.markdown("---")

    st.info("""
    **Optimal Chunk Size:**
    - Our chunks average ~300 characters
    - Balances context vs specificity
    - Fits well within embedding model limits
    - Enables precise retrieval
    """)

st.markdown("---")

# Sample chunks by gender
st.header("Sample Knowledge Chunks by Gender")

tab1, tab2, tab3 = st.tabs(
    ["👨‍✈️ Male Veteran Data", "👩‍✈️ Female Veteran Data", "🔷 Gender-Neutral Data"]
)

with tab1:
    male_chunks = rag.chunks_df[rag.chunks_df["gender_tag"] == "male"]
    if len(male_chunks) > 0:
        sample = male_chunks.sample(n=min(3, len(male_chunks)))
        for idx, row in sample.iterrows():
            st.markdown(f"""
            **Source:** {row["source"]}  
            **Length:** {len(row["text"])} characters
            
            **Text:**
            > {row["text"][:400]}{"..." if len(row["text"]) > 400 else ""}
            """)
            st.markdown("---")
    else:
        st.info("No male veteran-specific chunks found")

with tab2:
    female_chunks = rag.chunks_df[rag.chunks_df["gender_tag"] == "female"]
    if len(female_chunks) > 0:
        sample = female_chunks.sample(n=min(3, len(female_chunks)))
        for idx, row in sample.iterrows():
            st.markdown(f"""
            **Source:** {row["source"]}  
            **Length:** {len(row["text"])} characters
            
            **Text:**
            > {row["text"][:400]}{"..." if len(row["text"]) > 400 else ""}
            """)
            st.markdown("---")
    else:
        st.info("No female veteran-specific chunks found")

with tab3:
    neutral_chunks = rag.chunks_df[rag.chunks_df["gender_tag"] == "neutral"]
    if len(neutral_chunks) > 0:
        sample = neutral_chunks.sample(n=min(3, len(neutral_chunks)))
        for idx, row in sample.iterrows():
            st.markdown(f"""
            **Source:** {row["source"]}  
            **Length:** {len(row["text"])} characters
            
            **Text:**
            > {row["text"][:400]}{"..." if len(row["text"]) > 400 else ""}
            """)
            st.markdown("---")
    else:
        st.info("No gender-neutral chunks found")

st.markdown("---")

# System information
st.header("⚙️ System Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Configuration")
    st.code(f"""
Embedding Model: {rag.model_name}
Embedding Dimensions: {embedding_dim}
Index Type: FAISS IndexFlatIP
Similarity Metric: Inner Product (Cosine)
Total Vectors: {rag.faiss_index.ntotal:,}
Gender-Aware: Yes
    """)

with col2:
    st.subheader("Data Configuration")
    st.code(f"""
Corpus File: {rag.chunks_path.name}
Total Chunks: {len(rag.chunks_df):,}
Unique Sources: {sources_count}
Male Veteran Chunks: {gender_counts.get("male", 0):,}
Female Veteran Chunks: {gender_counts.get("female", 0):,}
Gender-Neutral Chunks: {gender_counts.get("neutral", 0):,}
Index Location: {rag.faiss_index_path}
    """)

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <strong>RAG System Statistics Dashboard</strong><br>
    Data Science Masters Project • Dave Singh & Nipu Quayum<br>
    Colorblind-Friendly Design • Accessible Visualizations
</div>
""",
    unsafe_allow_html=True,
)

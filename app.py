"""
Streamlit Chatbot Interface for Veteran Mental Health RAG System
Interactive web application for querying veteran mental health information

PRESENTATION MODE: Optimized for live demonstrations
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_system import VeteranHealthRAG

# Optional: Import Keras model if available
try:
    import tensorflow as tf
    from tensorflow import keras

    KERAS_AVAILABLE = True
except:
    KERAS_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Veteran Mental Health Assistant",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e78;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5a7a99;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1f4e78;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2d5f8f;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid;
    }
    .user-message {
        background-color: #e8f4f8;
        border-left-color: #1f4e78;
    }
    .assistant-message {
        background-color: #f0f2f6;
        border-left-color: #5a7a99;
    }
    .source-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system"""
    try:
        rag = VeteranHealthRAG(
            faiss_index_path="./models/faiss_index",
            chunks_path="./data/veteran_rag_corpus.csv",
        )

        try:
            # Try to load existing index
            rag.load_index()
            st.sidebar.success(" RAG system loaded from cache")
        except FileNotFoundError:
            # Build new index if doesn't exist
            st.sidebar.info("🔨 Building RAG system for first time...")
            rag.load_corpus()
            rag.create_embeddings(show_progress=False)
            rag.build_faiss_index(save=True)
            st.sidebar.success(" RAG system built and cached")

        # Test the system with a simple query
        try:
            test_result = rag.retrieve("test", k=1)
            if test_result is not None and len(test_result) > 0:
                st.sidebar.success(
                    f" System tested - {len(rag.chunks_df)} chunks ready"
                )
            else:
                st.sidebar.warning(" System loaded but test query returned no results")
        except Exception as e:
            st.sidebar.error(f" System test failed: {e}")
            st.error(f"Full error: {e}")
            import traceback

            st.code(traceback.format_exc())

        return rag

    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        import traceback

        st.code(traceback.format_exc())
        raise


def display_chat_message(
    role, content, sources=None, show_scores=False, highlight_gender=False
):
    """Display a chat message with optional sources and presentation features"""
    message_class = "user-message" if role == "user" else "assistant-message"
    icon = "🎖️" if role == "user" else "🤖"

    st.markdown(
        f"""
    <div class="chat-message {message_class}">
        <strong>{icon} {role.capitalize()}</strong><br><br>
        {content}
    </div>
    """,
        unsafe_allow_html=True,
    )

    if sources and role == "assistant":
        with st.expander(" View Sources & Evidence", expanded=False):
            for i, source in enumerate(sources, 1):
                # Highlight gender-specific sources
                gender_tag = ""
                if highlight_gender and "metadata" in source:
                    if isinstance(source.get("metadata"), dict):
                        gender = source["metadata"].get("gender", "")
                        if gender:
                            gender_tag = f" **[{gender.upper()} VETERAN DATA]**"

                source_display = f"""
                <div class="source-box">
                    <strong>Source {i}:</strong> {source["source"]}{gender_tag}<br>
                    <strong>Document:</strong> {source["doc_id"]}<br>
                    """

                if show_scores:
                    score = source.get("score", 0)
                    score_pct = score * 100
                    color = (
                        "green" if score > 0.5 else "orange" if score > 0.3 else "red"
                    )
                    source_display += f'<strong>Relevance Score:</strong> <span style="color: {color}; font-weight: bold; font-size: 1.1em;">{score:.3f} ({score_pct:.1f}%)</span><br>'

                source_display += "</div>"
                st.markdown(source_display, unsafe_allow_html=True)


def main():
    # Header
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #1f4e78 0%, #2d6a9f 100%); border-radius: 10px; margin-bottom: 1rem;'>
            <h2 style='color: white; margin: 0;'>Veteran Mental Health Assistant</h2>
            <p style='color: #e0e0e0; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Evidence-Based Support for Veterans & Families</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sub-header">AI-powered support for veterans, families, and healthcare providers</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎖️ Veteran Mental Health Assistant")

        st.markdown("### 🎓 Presentation Mode")
        presentation_mode = st.checkbox(
            "Enable Presentation Mode",
            value=True,
            help="Optimized for live demonstrations",
        )

        if presentation_mode:
            st.info(
                " **Presentation Tips:**\n\n"
                "• Use example questions below\n"
                "• Show source citations\n"
                "• Display system statistics\n"
                "• Explain retrieval scores"
            )

        st.markdown("### About This System")
        st.info("""
        This chatbot uses **Retrieval-Augmented Generation (RAG)** to provide 
        evidence-based information about veteran mental health, with a focus on 
        PTSD and related conditions.
        """)

        # Data Sources with links
        st.markdown("### Data Sources")
        with st.expander("View All Sources & Links", expanded=False):
            st.markdown("""
            **[BRFSS 2024 Survey Data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)**  
            *CDC Behavioral Risk Factor Surveillance System*
            
            **[Female Veteran Health Data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)**  
            *Processed from BRFSS 2024 - female veteran respondents*
            
            **[Male Veteran Health Data](https://www.cdc.gov/brfss/annual_data/annual_2024.html)**  
            *Processed from BRFSS 2024 - male veteran respondents*
            
            **[PTSD VA Sample Characteristics](https://ptsd-va.data.socrata.com/PTSD-Repository/Study-Characteristics/npcj-egem/about_data)**  
            *VA PTSD Research Repository*
            
            **[Medical Abbreviations](https://ptsd-va.data.socrata.com/PTSD-Repository/Abbreviations/46j5-9dq5/about_data)**  
            *VA PTSD Medical Terminology Database*
            
            **[PTSD Treatment Research](https://catalog.data.gov/dataset/ahrq-report-and-data-files-2023-pharmacological-and-nonpharmacological-treatments-for-post)**  
            *AHRQ Pharmacological & Nonpharmacological Treatments for PTSD*
            """)

        st.markdown("### ⚙️ Settings")

        num_sources = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="More sources provide better context but may be slower",
        )

        show_scores = st.checkbox(
            "Show relevance scores", value=True if presentation_mode else False
        )
        show_gender_analysis = st.checkbox("Highlight gender-specific data", value=True)

        use_llm = st.checkbox(
            "Use LLM Generation (requires API key)",
            value=False,
            help="Enable if you have OpenAI API key set",
        )

        if use_llm:
            llm_model = st.selectbox(
                "LLM Model", options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=0
            )
        else:
            llm_model = None

        # Optional Keras Model Integration
        if KERAS_AVAILABLE:
            st.markdown("### ML Model (Optional)")
            use_keras = st.checkbox(
                "Enable Keras Mental Health Classification", value=False
            )
            if use_keras:
                keras_model_path = st.text_input(
                    "Keras Model Path", value="./models/mental_health_model.h5"
                )
        else:
            use_keras = False

        st.markdown("### Example Questions")

        # Enhanced example questions for presentation
        example_categories = {
            "Symptoms & Diagnosis": [
                "What are common PTSD symptoms in veterans?",
                "How does combat exposure affect mental health?",
                "What are signs of depression in veterans?",
            ],
            "Treatment Options": [
                "What treatments are available for PTSD?",
                "What is Cognitive Processing Therapy?",
                "Are medications effective for veteran mental health?",
            ],
            "Gender-Specific": [
                "What mental health challenges do female veterans face?",
                "How does military sexual trauma affect mental health?",
                "Are there gender differences in PTSD rates?",
            ],
            "Support Resources": [
                "What support resources exist for veterans?",
                "How can families help veterans with PTSD?",
                "What is the Veterans Crisis Line?",
            ],
        }

        selected_category = st.selectbox(
            "Question Category:", options=list(example_categories.keys())
        )

        selected_example = st.selectbox(
            "Try an example:",
            options=[""] + example_categories[selected_category],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # System Statistics Button
        if st.button(" View System Statistics", width="stretch"):
            st.session_state.show_stats = True

        # Demo Mode
        if presentation_mode:
            st.markdown("---")
            st.markdown("### 🎬 Demo Mode")
            if st.button("Run Auto Demo", width="stretch"):
                st.session_state.run_demo = True

        st.markdown("---")
        st.markdown(
            """
        <div style='text-align: center; font-size: 0.8rem; color: #666;'>
            Built with RAG Technology<br>
            Data Science Masters Project<br>
            Dave Singh & Nipu Quayum
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Load RAG system
    try:
        rag = load_rag_system()
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        st.stop()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "query_count" not in st.session_state:
        st.session_state.query_count = 0

    # Show statistics if requested
    if st.session_state.get("show_stats", False):
        st.markdown("### System Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                """
            <div class="metric-card">
                <h3 style="color: #1f4e78; margin: 0;">"""
                + str(len(rag.chunks_df))
                + """</h3>
                <p style="margin: 0; color: #666;">Knowledge Chunks</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="metric-card">
                <h3 style="color: #1f4e78; margin: 0;">"""
                + str(rag.faiss_index.ntotal)
                + """</h3>
                <p style="margin: 0; color: #666;">Indexed Vectors</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
            <div class="metric-card">
                <h3 style="color: #1f4e78; margin: 0;">"""
                + str(st.session_state.query_count)
                + """</h3>
                <p style="margin: 0; color: #666;">Queries Answered</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            sources_count = len(rag.chunks_df["source"].unique())
            st.markdown(
                """
            <div class="metric-card">
                <h3 style="color: #1f4e78; margin: 0;">"""
                + str(sources_count)
                + """</h3>
                <p style="margin: 0; color: #666;">Data Sources</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Source distribution
        st.markdown("#### Data Source Distribution")
        source_counts = rag.chunks_df["source"].value_counts()

        # Show top sources in pie chart, group smaller ones
        top_n = 10
        top_sources = source_counts.head(top_n)
        other_count = source_counts[top_n:].sum()

        if other_count > 0:
            chart_data = pd.concat(
                [top_sources, pd.Series({"Other sources": other_count})]
            )
        else:
            chart_data = top_sources

        fig = px.pie(
            values=chart_data.values,
            names=chart_data.index,
            title=f"Chunks by Source (Top {top_n} + Other)",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )

        # Update layout - better positioning
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
            margin=dict(l=20, r=250, t=60, b=20),  # Large right margin for legend
            height=600,
        )

        # Show percentages on slices
        fig.update_traces(textposition="inside", textinfo="percent", textfont_size=12)

        st.plotly_chart(fig, use_container_width=True)

        # Show full list in expander
        with st.expander("📋 View All Data Sources"):
            st.dataframe(
                source_counts.reset_index().rename(
                    columns={"index": "Source", "source": "Number of Chunks"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        if st.button("Close Statistics"):
            st.session_state.show_stats = False
            st.rerun()

        st.markdown("---")

    # Auto Demo Mode (for presentation)
    if st.session_state.get("run_demo", False):
        st.info("🎬 **Running Auto Demo** - Demonstrating key queries...")

        demo_queries = [
            "What are common PTSD symptoms in veterans?",
            "What mental health challenges do female veterans face?",
            "What treatments are available for PTSD?",
        ]

        for query in demo_queries:
            st.markdown(f"### Demo Query: {query}")

            with st.spinner(f"Processing: {query}"):
                try:
                    result = rag.answer_query(query=query, k=3, use_llm=False)

                    st.markdown(f"**Answer:**\n{result['answer']}")

                    with st.expander("Sources"):
                        for i, source in enumerate(result["sources"], 1):
                            st.write(
                                f"{i}. {source['source']} (score: {source['score']:.3f})"
                            )

                    time.sleep(2)  # Pause between queries

                except Exception as e:
                    st.error(f"Demo error: {e}")

            st.markdown("---")

        st.success(" Demo Complete!")
        st.session_state.run_demo = False
        st.rerun()
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("sources"),
            show_scores=show_scores,
            highlight_gender=show_gender_analysis,
        )

    # Disclaimer
    st.markdown(
        """
    <div class="disclaimer">
        <strong>⚠️ Important Disclaimer:</strong><br>
        This chatbot provides educational information only and is not a substitute for 
        professional medical advice, diagnosis, or treatment. If you are experiencing a 
        mental health emergency, please contact the <strong>Veterans Crisis Line</strong> 
        at <strong>988 (Press 1)</strong> or text <strong>838255</strong>.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Footer actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Clear Chat History", key="clear_history_btn"):
            # Clear all chat-related state
            st.session_state.messages = []
            st.session_state.query_count = 0

            # Force clear any cached data
            if "last_query" in st.session_state:
                del st.session_state["last_query"]
            if "last_response" in st.session_state:
                del st.session_state["last_response"]

            # Clear Streamlit cache
            st.cache_data.clear()

            st.rerun()

    with col2:
        if st.button("💾 Export Conversation"):
            # Create export data
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "query_count": st.session_state.query_count,
            }

            st.download_button(
                label="Download JSON",
                data=str(export_data),
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    with col3:
        st.markdown("")  # Placeholder

    # Chat input
    user_input = st.chat_input("Ask a question about veteran mental health...")

    # Handle example question selection - only if no manual input
    if not user_input and selected_example and selected_example != "":
        user_input = selected_example
        # Clear the example selection after using it
        st.session_state.used_example = True

    # Process user input
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        display_chat_message("user", user_input)

        # Generate response
        with st.spinner("🔍 Searching knowledge base and generating response..."):
            try:
                result = rag.answer_query(
                    query=user_input,
                    k=num_sources,
                    use_llm=use_llm,
                    llm_model=llm_model if use_llm else None,
                )

                # Add assistant message to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )

                # Display assistant message
                display_chat_message(
                    "assistant",
                    result["answer"],
                    result["sources"],
                    show_scores=show_scores,
                    highlight_gender=show_gender_analysis,
                )

                # Update query count
                st.session_state.query_count += 1

            except Exception as e:
                st.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()

"""
Enhanced RAG System for Veteran Mental Health Chatbot
Extends midterm RAG work with dense embeddings, FAISS, and domain-specific features
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VeteranHealthRAG:
    """
    RAG system specialized for veteran mental health domain
    Combines retrieval with semantic understanding
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        faiss_index_path: str = "./models/faiss_index",
        chunks_path: str = "./data/veteran_rag_corpus.csv",
    ):
        """
        Initialize RAG system

        Args:
            model_name: HuggingFace model for embeddings
            faiss_index_path: Path to save/load FAISS index
            chunks_path: Path to chunked corpus CSV
        """
        self.model_name = model_name
        self.faiss_index_path = Path(faiss_index_path)
        self.chunks_path = Path(chunks_path)

        # Initialize components
        self.encoder = None
        self.chunks_df = None
        self.faiss_index = None
        self.embeddings = None

        # Mental health keywords for domain-specific boosting
        self.mental_health_keywords = [
            "ptsd",
            "trauma",
            "anxiety",
            "depression",
            "stress",
            "therapy",
            "treatment",
            "veteran",
            "military",
            "combat",
            "mental health",
            "psychological",
            "psychiatric",
            "counseling",
            "suicide",
            "substance",
            "alcohol",
            "medication",
            "diagnosis",
        ]

    def load_corpus(self, chunks_path: Optional[str] = None):
        """
        Load pre-processed chunks

        Args:
            chunks_path: Path to chunks CSV (optional override)
        """
        if chunks_path:
            self.chunks_path = Path(chunks_path)

        logger.info(f"Loading corpus from {self.chunks_path}")

        try:
            self.chunks_df = pd.read_csv(self.chunks_path)
            logger.info(f"Loaded {len(self.chunks_df)} chunks")

            # Convert string metadata back to dict if needed
            if "metadata" in self.chunks_df.columns:
                import ast

                self.chunks_df["metadata"] = self.chunks_df["metadata"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )

            return self.chunks_df

        except FileNotFoundError:
            logger.error(f"Corpus file not found: {self.chunks_path}")
            raise

    def initialize_encoder(self):
        """Initialize sentence transformer model for embeddings"""
        logger.info(f"Loading encoder model: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)
        logger.info("Encoder loaded successfully")

    def create_embeddings(self, show_progress: bool = True):
        """
        Create dense embeddings for all chunks

        Args:
            show_progress: Show progress bar during encoding
        """
        if self.chunks_df is None:
            raise ValueError("Corpus not loaded. Call load_corpus() first")

        if self.encoder is None:
            self.initialize_encoder()

        logger.info("Creating embeddings for all chunks...")

        texts = self.chunks_df["text"].tolist()

        self.embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ).astype("float32")

        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")

        return self.embeddings

    def build_faiss_index(self, save: bool = True):
        """
        Build FAISS index for fast similarity search

        Args:
            save: Whether to save index to disk
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Call create_embeddings() first")

        logger.info("Building FAISS index...")

        # Use Inner Product (IP) for normalized embeddings (equivalent to cosine)
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings)

        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")

        if save:
            self.save_index()

    def save_index(self):
        """Save FAISS index and metadata to disk"""
        self.faiss_index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = self.faiss_index_path / "index.faiss"
        faiss.write_index(self.faiss_index, str(index_file))
        logger.info(f"Saved FAISS index to {index_file}")

        # Save metadata (chunks dataframe and embeddings info)
        metadata = {
            "chunks_df": self.chunks_df,
            "model_name": self.model_name,
            "embedding_dim": self.embeddings.shape[1],
            "num_chunks": len(self.chunks_df),
        }

        metadata_file = self.faiss_index_path / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_file}")

    def load_index(self):
        """Load FAISS index and metadata from disk"""
        index_file = self.faiss_index_path / "index.faiss"
        metadata_file = self.faiss_index_path / "metadata.pkl"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Index files not found in {self.faiss_index_path}")

        # Load FAISS index
        self.faiss_index = faiss.read_index(str(index_file))
        logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

        # Load metadata
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)

        self.chunks_df = metadata["chunks_df"]
        logger.info(f"Loaded metadata for {len(self.chunks_df)} chunks")

        # Initialize encoder if needed
        if self.encoder is None:
            self.initialize_encoder()

    def retrieve(
        self, query: str, k: int = 5, boost_domain_keywords: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve top-k most relevant chunks for a query

        Args:
            query: User query string
            k: Number of chunks to retrieve
            boost_domain_keywords: Boost scores for mental health domain terms

        Returns:
            DataFrame with retrieved chunks and scores
        """
        if self.encoder is None:
            self.initialize_encoder()

        if self.faiss_index is None:
            raise ValueError(
                "FAISS index not built. Call build_faiss_index() or load_index()"
            )

        # Encode query
        query_embedding = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        logger.debug(f"Query embedding type: {type(query_embedding)}")

        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, k)

        logger.debug(
            f"Scores type: {type(scores)}, shape: {scores.shape if hasattr(scores, 'shape') else 'N/A'}"
        )
        logger.debug(
            f"Indices type: {type(indices)}, shape: {indices.shape if hasattr(indices, 'shape') else 'N/A'}"
        )
        logger.debug(f"Scores: {scores}")
        logger.debug(f"Indices: {indices}")

        # Handle single dimension arrays
        if len(scores.shape) == 1:
            logger.debug("Reshaping 1D arrays to 2D")
            scores = scores.reshape(1, -1)
            indices = indices.reshape(1, -1)

        # Extract first row (we only query one at a time)
        try:
            scores = scores[0]
            indices = indices[0]
            logger.debug(f"After extraction - Scores: {scores}, Indices: {indices}")
        except Exception as e:
            logger.error(f"Error extracting scores/indices: {e}")
            logger.error(f"scores type: {type(scores)}, value: {scores}")
            logger.error(f"indices type: {type(indices)}, value: {indices}")
            raise

        # Build results dataframe
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            # Ensure idx is an integer
            try:
                idx = int(idx)
            except Exception as e:
                logger.error(
                    f"Error converting idx to int: {e}, idx={idx}, type={type(idx)}"
                )
                continue

            # Safely get chunk data
            if idx >= len(self.chunks_df):
                logger.warning(f"Index {idx} out of bounds, skipping")
                continue

            chunk_data = self.chunks_df.iloc[idx].to_dict()
            chunk_data["rank"] = rank + 1

            # Convert score to float safely
            try:
                chunk_data["score"] = float(score)
            except Exception as e:
                logger.error(
                    f"Error converting score to float: {e}, score={score}, type={type(score)}"
                )
                chunk_data["score"] = 0.0

            # Boost score if mental health keywords present
            if boost_domain_keywords:
                text_lower = str(chunk_data.get("text", "")).lower()
                keyword_count = sum(
                    1 for kw in self.mental_health_keywords if kw in text_lower
                )
                if keyword_count > 0:
                    boost = min(0.1, keyword_count * 0.02)  # Max 10% boost
                    chunk_data["score"] += boost
                    chunk_data["boosted"] = True
                else:
                    chunk_data["boosted"] = False

            results.append(chunk_data)

        if not results:
            logger.warning("No results found!")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Re-sort by boosted scores if applicable
        if boost_domain_keywords:
            results_df = results_df.sort_values("score", ascending=False).reset_index(
                drop=True
            )
            results_df["rank"] = range(1, len(results_df) + 1)

        return results_df

    def build_context_prompt(
        self, query: str, retrieved_chunks: pd.DataFrame, max_context_length: int = 4000
    ) -> Tuple[str, List[Dict]]:
        """
        Build context-aware prompt for LLM generation

        Args:
            query: User query
            retrieved_chunks: DataFrame of retrieved chunks
            max_context_length: Maximum characters for context

        Returns:
            Tuple of (prompt, citations)
        """
        context_parts = []
        citations = []
        total_chars = 0

        for _, row in retrieved_chunks.iterrows():
            # Format citation
            citation = {
                "source": row["source"],
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"],
                "score": row["score"],
            }

            # Format context piece
            context_piece = (
                f"\n[Source: {row['source']} - {row['chunk_id']}]\n{row['text']}\n"
            )

            # Check if adding this would exceed limit
            if total_chars + len(context_piece) > max_context_length:
                break

            context_parts.append(context_piece)
            citations.append(citation)
            total_chars += len(context_piece)

        context = "\n".join(context_parts)

        # Build prompt
        prompt = f"""You are a knowledgeable assistant specializing in veteran mental health, particularly PTSD and related conditions. 
You provide evidence-based information to help veterans, their families, and healthcare providers.

Answer the user's question using ONLY the context provided below. 
If the answer is not in the context, say "I don't have enough information in my knowledge base to answer that question accurately."
Always cite your sources using [Source: ...] notation.

Be empathetic and supportive in your responses, recognizing the sensitive nature of mental health topics.

Question: {query}

Context:
{context}

Answer:"""

        return prompt, citations

    def answer_query(
        self,
        query: str,
        k: int = 5,
        use_llm: bool = False,
        llm_model: str = "gpt-3.5-turbo",
    ) -> Dict:
        """
        End-to-end RAG: retrieve context and generate answer

        Args:
            query: User query
            k: Number of chunks to retrieve
            use_llm: Whether to use LLM for generation
            llm_model: Which LLM to use (if use_llm=True)

        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, k=k)

        # Build prompt and get citations
        prompt, citations = self.build_context_prompt(query, retrieved)

        # Generate answer
        if use_llm:
            try:
                import os

                import openai

                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a veteran mental health assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )

                answer = response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(
                    f"LLM generation failed: {e}. Using extractive fallback."
                )
                answer = self._extractive_answer(query, retrieved)
        else:
            # Use extractive summarization
            answer = self._extractive_answer(query, retrieved)

        return {
            "query": query,
            "answer": answer,
            "sources": citations,
            "num_sources": len(citations),
            "top_score": float(retrieved.iloc[0]["score"])
            if len(retrieved) > 0
            else 0.0,
        }

    def _extractive_answer(
        self, query: str, retrieved_chunks: pd.DataFrame, max_sentences: int = 3
    ) -> str:
        """
        Create answer by extracting and formatting information from retrieved chunks

        Args:
            query: User query
            retrieved_chunks: Retrieved chunks
            max_sentences: Maximum sentences to include

        Returns:
            Formatted answer string
        """
        # Check if this is a treatment-related query
        treatment_keywords = [
            "treatment",
            "therapy",
            "medication",
            "intervention",
            "available",
            "options",
        ]
        is_treatment_query = any(kw in query.lower() for kw in treatment_keywords)

        if is_treatment_query and "Treatment:" in retrieved_chunks["text"].iloc[0]:
            # Format as treatment list
            return self._format_treatment_answer(query, retrieved_chunks)
        else:
            # Use standard extractive approach
            return self._format_standard_answer(query, retrieved_chunks, max_sentences)

    def _format_treatment_answer(
        self, query: str, retrieved_chunks: pd.DataFrame
    ) -> str:
        """Format treatment information in a clean, readable way"""
        treatments = {}

        for _, row in retrieved_chunks.head(10).iterrows():
            text = row["text"]

            # Parse treatment info
            treatment_name = None
            treatment_type = None
            description = None
            category = None
            details = []

            parts = text.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("Treatment:"):
                    treatment_name = part.replace("Treatment:", "").strip()
                elif part.startswith("Type:"):
                    treatment_type = part.replace("Type:", "").strip()
                elif part.startswith("Description:"):
                    description = part.replace("Description:", "").strip()
                elif part.startswith("Category:"):
                    category = part.replace("Category:", "").strip()
                elif part.startswith("Duration:"):
                    details.append(part)
                elif part.startswith("Format:"):
                    details.append(part)
                elif part.startswith("Delivery:"):
                    details.append(part)
                elif part.startswith("Completion Rate:"):
                    details.append(part)

            if treatment_name and treatment_name not in treatments:
                treatments[treatment_name] = {
                    "type": treatment_type,
                    "description": description,
                    "category": category,
                    "details": details,
                }

        # Build formatted answer
        answer_parts = []
        answer_parts.append("**Evidence-Based PTSD Treatments:**\n")

        for i, (name, info) in enumerate(list(treatments.items())[:5], 1):
            answer_parts.append(f"\n**{i}. {name}**")

            if info["category"]:
                answer_parts.append(f"\n- **Type:** {info['category']}")

            if info["description"]:
                # Limit description length
                desc = info["description"]
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                answer_parts.append(f"\n- **Description:** {desc}")

            if info["details"]:
                detail_str = ", ".join(d for d in info["details"][:3])
                answer_parts.append(f"\n- **Details:** {detail_str}")

            answer_parts.append("")  # Blank line

        # Add source attribution with top scores
        sources = retrieved_chunks["source"].unique()[:3]
        top_scores = retrieved_chunks.head(3)["score"].tolist()

        source_list = []
        for i, (src, score) in enumerate(zip(sources, top_scores)):
            source_list.append(f"{src} ({score:.3f})")

        answer_parts.append(f"\n**Sources:** {', '.join(source_list)}")

        return "\n".join(answer_parts)

    def _format_standard_answer(
        self, query: str, retrieved_chunks: pd.DataFrame, max_sentences: int = 3
    ) -> str:
        """Standard extractive answer formatting with improved readability"""
        # Combine all retrieved text
        combined_text = " ".join(retrieved_chunks["text"].tolist())

        # Check if text contains numbered lists like (1), (2), (3)
        import re

        has_numbered_parens = bool(re.search(r"\((\d+)\)", combined_text))

        if has_numbered_parens:
            # Extract and format numbered list items
            items = re.split(r"\((\d+)\)", combined_text)
            formatted_items = []

            for i in range(1, len(items), 2):
                if i + 1 < len(items):
                    num = items[i]
                    text = items[i + 1].strip()
                    # Clean up the text
                    text = text.split(";")[
                        0
                    ].strip()  # Take first part before semicolon
                    if text:
                        formatted_items.append(f"{num}. {text}")

            if formatted_items:
                answer = "**Key Points:**\n\n" + "\n\n".join(
                    formatted_items[:max_sentences]
                )

                # Add source attribution with scores
                sources = retrieved_chunks["source"].unique()[:3]
                top_scores = retrieved_chunks.head(3)["score"].tolist()

                source_list = []
                for src, score in zip(sources, top_scores):
                    source_list.append(f"{src} ({score:.3f})")

                answer += f"\n\n---\n**Sources:** {', '.join(source_list)}"
                return answer

        # Split into sentences
        sentences = [s.strip() for s in combined_text.split(".") if len(s.strip()) > 20]

        # Score sentences by keyword overlap with query
        query_terms = set(query.lower().split())

        def score_sentence(sent):
            sent_terms = set(sent.lower().split())
            overlap = len(query_terms & sent_terms)
            # Boost for mental health keywords
            domain_boost = sum(
                1 for kw in self.mental_health_keywords if kw in sent.lower()
            )
            return overlap + (domain_boost * 0.5)

        # Get top sentences
        scored_sentences = [(score_sentence(s), s) for s in sentences]
        scored_sentences.sort(reverse=True, key=lambda x: x[0])

        top_sentences = [s for _, s in scored_sentences[:max_sentences]]

        # Format with better structure
        if len(top_sentences) == 1:
            # Single sentence - just display it
            answer = top_sentences[0]
            if not answer.endswith("."):
                answer += "."
        else:
            # Multiple sentences - format as numbered list or paragraphs
            # Check if content looks like criteria/list items
            if any(
                keyword in combined_text.lower()
                for keyword in [
                    "exclusion criteria",
                    "inclusion criteria",
                    "following:",
                    "criteria were",
                ]
            ):
                # Format as bullet points for criteria/lists
                answer = "**Key Information:**\n\n"
                for i, sent in enumerate(top_sentences, 1):
                    if not sent.endswith("."):
                        sent += "."
                    answer += f"• {sent}\n\n"
            else:
                # Format as paragraphs with spacing
                formatted_sentences = []
                for sent in top_sentences:
                    if not sent.endswith("."):
                        sent += "."
                    formatted_sentences.append(sent)
                answer = "\n\n".join(formatted_sentences)

        # Add source attribution with scores
        sources = retrieved_chunks["source"].unique()[:3]
        top_scores = retrieved_chunks.head(3)["score"].tolist()

        source_list = []
        for src, score in zip(sources, top_scores):
            source_list.append(f"{src} ({score:.3f})")

        answer += f"\n\n---\n**Sources:** {', '.join(source_list)}"

        return answer


def main():
    """Example usage of VeteranHealthRAG"""

    # Initialize RAG system
    rag = VeteranHealthRAG(
        faiss_index_path="./models/faiss_index",
        chunks_path="./data/veteran_rag_corpus.csv",
    )

    try:
        # Load corpus
        rag.load_corpus()

        # Create embeddings and build index
        rag.create_embeddings()
        rag.build_faiss_index(save=True)

        # Test queries
        test_queries = [
            "What are the common symptoms of PTSD in veterans?",
            "What treatments are available for veteran mental health?",
            "How does military service affect mental health?",
        ]

        print("\n" + "=" * 80)
        print("Testing Veteran Health RAG System")
        print("=" * 80)

        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 80)

            result = rag.answer_query(query, k=3, use_llm=False)

            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nNumber of sources used: {result['num_sources']}")
            print(f"Top relevance score: {result['top_score']:.3f}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()

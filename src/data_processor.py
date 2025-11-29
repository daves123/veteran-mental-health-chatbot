"""
Data Processing Module for Veteran Mental Health Chatbot
Handles loading, cleaning, and preparing BRFSS, PTSD, and research datasets
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VeteranDataProcessor:
    """Process and prepare veteran mental health datasets for RAG system"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.processed_chunks = None

    def load_sample_characteristics(self, filepath: str = None) -> pd.DataFrame:
        """
        Load PTSD VA Sample Characteristics dataset

        Args:
            filepath: Path to Sample_Characteristics CSV file

        Returns:
            DataFrame with sample characteristics data
        """
        if filepath is None:
            filepath = self.data_dir / "Sample_Characteristics_20251123.csv"

        logger.info(f"Loading Sample Characteristics from {filepath}")

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from Sample Characteristics")
            self.datasets["sample_characteristics"] = df
            return df
        except Exception as e:
            logger.error(f"Error loading Sample Characteristics: {e}")
            raise

    def load_brfss_2024(
        self, filepath: str = None, sample_size: int = 10000
    ) -> pd.DataFrame:
        """
        Load BRFSS 2024 data from XPT (SAS transport) file

        Args:
            filepath: Path to LLCP2024.XPT file
            sample_size: Load only first N rows (default 10000, None = all)

        Returns:
            DataFrame with BRFSS 2024 data
        """
        if filepath is None:
            filepath = self.data_dir / "LLCP2024.XPT"

        logger.info(f"Loading BRFSS 2024 from {filepath}")

        try:
            # Read XPT file (SAS transport format)
            if sample_size:
                logger.info(f"Loading sample of {sample_size} records (file is 1GB)")
                # Use chunksize for memory efficiency
                chunks = []
                for chunk in pd.read_sas(filepath, format="xport", chunksize=5000):
                    chunks.append(chunk)
                    if len(pd.concat(chunks)) >= sample_size:
                        break
                df = pd.concat(chunks).head(sample_size)
            else:
                logger.info("Loading full BRFSS dataset (may take several minutes...)")
                df = pd.read_sas(filepath, format="xport")

            logger.info(f"Loaded {len(df)} records from BRFSS 2024")
            logger.info(f"Columns: {len(df.columns)}")
            self.datasets["brfss_2024"] = df
            return df
        except Exception as e:
            logger.error(f"Error loading BRFSS 2024: {e}")
            logger.info("Tip: Install required package with: pip install pandas[sas]")
            raise

    def load_veteran_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load pre-cleaned veteran datasets (female and male)

        Returns:
            Dictionary with female and male veteran DataFrames
        """
        veteran_data = {}

        # Load female veterans
        female_path = self.data_dir / "female_veterans_clean.csv"
        if female_path.exists():
            logger.info(f"Loading female veterans data from {female_path}")
            try:
                df = pd.read_csv(female_path)
                logger.info(f"Loaded {len(df)} female veteran records")
                veteran_data["female_veterans"] = df
                self.datasets["female_veterans"] = df
            except Exception as e:
                logger.warning(f"Could not load female veterans data: {e}")

        # Load male veterans
        male_path = self.data_dir / "male_veterans_clean.csv"
        if male_path.exists():
            logger.info(f"Loading male veterans data from {male_path}")
            try:
                df = pd.read_csv(male_path)
                logger.info(f"Loaded {len(df)} male veteran records")
                veteran_data["male_veterans"] = df
                self.datasets["male_veterans"] = df
            except Exception as e:
                logger.warning(f"Could not load male veterans data: {e}")

        # Load mental health statistics summary
        summary_path = self.data_dir / "mental_health_statistics_summary.csv"
        if summary_path.exists():
            logger.info(f"Loading mental health statistics summary")
            try:
                df = pd.read_csv(summary_path)
                logger.info(f"Loaded mental health statistics summary")
                veteran_data["mh_summary"] = df
                self.datasets["mh_summary"] = df
            except Exception as e:
                logger.warning(f"Could not load mental health summary: {e}")

        return veteran_data

    def load_abbreviations(self, filepath: str = None) -> pd.DataFrame:
        """
        Load abbreviations/codebook for interpreting data

        Args:
            filepath: Path to Abbreviations CSV file

        Returns:
            DataFrame with abbreviations
        """
        if filepath is None:
            filepath = self.data_dir / "Abbreviations_20251123.csv"

        logger.info(f"Loading Abbreviations from {filepath}")

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} abbreviation records")
            self.datasets["abbreviations"] = df
            return df
        except Exception as e:
            logger.error(f"Error loading Abbreviations: {e}")
            raise

    def load_ptsd_research_data(self, data_dir: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load PTSD research Excel files

        Args:
            data_dir: Directory containing pharma-nonpharma PTSD files

        Returns:
            Dictionary of DataFrames for each Excel file
        """
        if data_dir is None:
            data_dir = self.data_dir / "pharma-nonpharma-ptsd-2023-update-app-g-2"

        data_dir = Path(data_dir)
        logger.info(f"Loading PTSD research data from {data_dir}")

        excel_files = list(data_dir.glob("*.xlsx"))
        ptsd_data = {}

        for file in excel_files:
            try:
                logger.info(f"Loading {file.name}")
                df = pd.read_excel(file)
                key = file.stem
                ptsd_data[key] = df
                logger.info(f"Loaded {len(df)} records from {file.name}")
            except Exception as e:
                logger.warning(f"Could not load {file.name}: {e}")

        self.datasets["ptsd_research"] = ptsd_data
        return ptsd_data

    def create_text_chunks(
        self, chunk_size: int = 300, overlap: int = 50
    ) -> pd.DataFrame:
        """
        Convert datasets into text chunks for RAG retrieval

        Args:
            chunk_size: Number of words per chunk
            overlap: Number of overlapping words between chunks

        Returns:
            DataFrame with columns: doc_id, chunk_id, text, source, metadata
        """
        logger.info("Creating text chunks for RAG system")

        all_chunks = []
        chunk_counter = 0

        # Process Female Veterans data (PRIORITY - for personal motivation)
        if "female_veterans" in self.datasets:
            df = self.datasets["female_veterans"]
            logger.info(f"Processing {len(df)} female veteran records")
            for idx, row in df.iterrows():
                text_parts = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        text_parts.append(f"{col}: {row[col]}")

                text = ". ".join(text_parts)
                chunks = self._chunk_text(text, chunk_size, overlap)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(
                        {
                            "doc_id": "female_veterans",
                            "chunk_id": f"female_vet_row{idx}_chunk{i}",
                            "text": chunk,
                            "source": "Female Veterans Health Data (BRFSS)",
                            "metadata": {
                                "row_index": idx,
                                "chunk_index": i,
                                "gender": "female",
                            },
                        }
                    )
                    chunk_counter += 1

        # Process Male Veterans data
        if "male_veterans" in self.datasets:
            df = self.datasets["male_veterans"]
            logger.info(f"Processing {len(df)} male veteran records")
            # Sample to avoid overwhelming the system
            sample_df = df.sample(min(len(df), 5000)) if len(df) > 5000 else df
            for idx, row in sample_df.iterrows():
                text_parts = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        text_parts.append(f"{col}: {row[col]}")

                text = ". ".join(text_parts)
                chunks = self._chunk_text(text, chunk_size, overlap)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(
                        {
                            "doc_id": "male_veterans",
                            "chunk_id": f"male_vet_row{idx}_chunk{i}",
                            "text": chunk,
                            "source": "Male Veterans Health Data (BRFSS)",
                            "metadata": {
                                "row_index": idx,
                                "chunk_index": i,
                                "gender": "male",
                            },
                        }
                    )
                    chunk_counter += 1

        # Process Mental Health Summary Statistics
        if "mh_summary" in self.datasets:
            df = self.datasets["mh_summary"]
            for idx, row in df.iterrows():
                text_parts = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        text_parts.append(f"{col}: {row[col]}")

                text = ". ".join(text_parts)

                all_chunks.append(
                    {
                        "doc_id": "mh_summary",
                        "chunk_id": f"mh_summary_{idx}",
                        "text": text,
                        "source": "Mental Health Statistics Summary",
                        "metadata": {"row_index": idx},
                    }
                )
                chunk_counter += 1

        # Process Sample Characteristics
        if "sample_characteristics" in self.datasets:
            df = self.datasets["sample_characteristics"]
            for idx, row in df.iterrows():
                text_parts = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        text_parts.append(f"{col}: {row[col]}")

                text = ". ".join(text_parts)
                chunks = self._chunk_text(text, chunk_size, overlap)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(
                        {
                            "doc_id": "sample_characteristics",
                            "chunk_id": f"sample_char_row{idx}_chunk{i}",
                            "text": chunk,
                            "source": "PTSD VA Sample Characteristics",
                            "metadata": {"row_index": idx, "chunk_index": i},
                        }
                    )
                    chunk_counter += 1

        # Process PTSD Research Data
        if "ptsd_research" in self.datasets:
            for doc_name, df in self.datasets["ptsd_research"].items():
                for idx, row in df.iterrows():
                    text_parts = []
                    for col in df.columns:
                        if pd.notna(row[col]):
                            text_parts.append(f"{col}: {row[col]}")

                    text = ". ".join(text_parts)
                    chunks = self._chunk_text(text, chunk_size, overlap)

                    for i, chunk in enumerate(chunks):
                        all_chunks.append(
                            {
                                "doc_id": doc_name,
                                "chunk_id": f"{doc_name}_row{idx}_chunk{i}",
                                "text": chunk,
                                "source": f"PTSD Research: {doc_name}",
                                "metadata": {
                                    "row_index": idx,
                                    "chunk_index": i,
                                    "document": doc_name,
                                },
                            }
                        )
                        chunk_counter += 1

        # Process Abbreviations as knowledge base
        if "abbreviations" in self.datasets:
            df = self.datasets["abbreviations"]
            for idx, row in df.iterrows():
                text_parts = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        text_parts.append(f"{col}: {row[col]}")

                text = ". ".join(text_parts)

                all_chunks.append(
                    {
                        "doc_id": "abbreviations",
                        "chunk_id": f"abbrev_{idx}",
                        "text": text,
                        "source": "Medical Abbreviations",
                        "metadata": {"row_index": idx},
                    }
                )
                chunk_counter += 1

        # Process BRFSS 2024 (if loaded - sample only)
        if "brfss_2024" in self.datasets:
            df = self.datasets["brfss_2024"]
            logger.info(f"Processing {len(df)} BRFSS 2024 records")
            # Focus on veteran-related and mental health variables
            for idx, row in df.iterrows():
                text_parts = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        text_parts.append(f"{col}: {row[col]}")

                text = ". ".join(text_parts)
                chunks = self._chunk_text(text, chunk_size, overlap)

                for i, chunk in enumerate(chunks):
                    all_chunks.append(
                        {
                            "doc_id": "brfss_2024",
                            "chunk_id": f"brfss2024_row{idx}_chunk{i}",
                            "text": chunk,
                            "source": "BRFSS 2024 Survey Data",
                            "metadata": {"row_index": idx, "chunk_index": i},
                        }
                    )
                    chunk_counter += 1

        logger.info(f"Created {chunk_counter} text chunks from all datasets")

        self.processed_chunks = pd.DataFrame(all_chunks)
        return self.processed_chunks

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Input text to chunk
            chunk_size: Number of words per chunk
            overlap: Number of overlapping words

        Returns:
            List of text chunks
        """
        words = text.split()

        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)

            if end == len(words):
                break

            start = max(0, end - overlap)

        return chunks

    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about loaded datasets

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "datasets_loaded": list(self.datasets.keys()),
            "total_datasets": len(self.datasets),
        }

        if "sample_characteristics" in self.datasets:
            df = self.datasets["sample_characteristics"]
            stats["sample_characteristics"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
            }

        if "ptsd_research" in self.datasets:
            ptsd_stats = {}
            for name, df in self.datasets["ptsd_research"].items():
                ptsd_stats[name] = {"rows": len(df), "columns": len(df.columns)}
            stats["ptsd_research"] = ptsd_stats

        if self.processed_chunks is not None:
            stats["chunks"] = {
                "total_chunks": len(self.processed_chunks),
                "sources": self.processed_chunks["source"].unique().tolist(),
            }

        return stats

    def export_chunks(self, filepath: str = "./data/veteran_rag_corpus.csv"):
        """
        Export processed chunks to CSV

        Args:
            filepath: Path to save CSV file
        """
        if self.processed_chunks is None:
            raise ValueError("No chunks to export. Run create_text_chunks() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert metadata dict to string for CSV storage
        df_export = self.processed_chunks.copy()
        df_export["metadata"] = df_export["metadata"].apply(str)

        df_export.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df_export)} chunks to {filepath}")


def main():
    """Example usage of VeteranDataProcessor with actual datasets"""

    # Initialize processor
    processor = VeteranDataProcessor(data_dir="./data")

    # Load datasets
    try:
        print("=" * 80)
        print("LOADING VETERAN MENTAL HEALTH DATASETS")
        print("=" * 80)

        # Load pre-cleaned veteran datasets (PRIORITY)
        print("\n1. Loading Pre-Cleaned Veteran Data...")
        veteran_data = processor.load_veteran_datasets()
        if veteran_data:
            for key, df in veteran_data.items():
                print(f"  ✓ {key}: {len(df)} records")

        # Load VA Sample Characteristics
        print("\n2. Loading PTSD VA Sample Characteristics...")
        processor.load_sample_characteristics()
        print(f"  ✓ Loaded {len(processor.datasets['sample_characteristics'])} records")

        # Load Abbreviations
        print("\n3. Loading Medical Abbreviations...")
        processor.load_abbreviations()
        print(f"  ✓ Loaded {len(processor.datasets['abbreviations'])} abbreviations")

        # Load PTSD Research
        print("\n4. Loading PTSD Treatment Research Data...")
        processor.load_ptsd_research_data()
        for name, df in processor.datasets["ptsd_research"].items():
            print(f"  ✓ {name}: {len(df)} records")

        # Optional: Load BRFSS 2024 (sample only - file is 1GB)
        print("\n5. BRFSS 2024 Data (Optional - loading sample)...")
        print("  Note: Full file is 1GB. Loading 10,000 records sample...")
        try:
            processor.load_brfss_2024(sample_size=10000)
            print(f"  ✓ Loaded {len(processor.datasets['brfss_2024'])} BRFSS records")
        except Exception as e:
            print(f"  ⚠ BRFSS 2024 skipped: {e}")
            print("  (This is optional - you have plenty of data without it)")

        # Get statistics
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)
        stats = processor.get_dataset_statistics()
        for key, value in stats.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f" {k}: {v}")
            else:
                print(f" {value}")

        # Create chunks for RAG
        print("\n" + "=" * 80)
        print("CREATING TEXT CHUNKS FOR RAG")
        print("=" * 80)
        chunks_df = processor.create_text_chunks(chunk_size=300, overlap=50)
        print(f"\n✓ Created {len(chunks_df)} chunks for RAG system")

        # Show chunk distribution by source
        print("\nChunk Distribution by Source:")
        source_counts = chunks_df["source"].value_counts()
        for source, count in source_counts.items():
            print(f" {source}: {count} chunks")

        print("\nSample chunks:")
        print(chunks_df[["source", "doc_id", "chunk_id"]].head(10))

        # Export chunks
        output_path = "./data/veteran_rag_corpus.csv"
        processor.export_chunks(output_path)
        print(f"\n✓ Exported chunks to {output_path}")

        print("\n" + "=" * 80)
        print("✓ DATA PROCESSING COMPLETE!")
        print("=" * 80)
        print("\nYour data is ready for the RAG system!")
        print("Next steps:")
        print(" 1. Run the Jupyter notebook for EDA and analysis")
        print(" 2. Build the FAISS index")
        print(" 3. Launch the Streamlit chatbot")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

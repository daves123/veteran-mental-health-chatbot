"""
Test Script for Veteran Mental Health RAG Enrichment
Verifies corpus enrichment and answer quality improvements
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd

from rag_system import VeteranHealthRAG


def test_corpus_enrichment():
    """Test that corpus has been enriched with PTSD content"""
    print("=" * 80)
    print("TEST 1: Corpus Enrichment Verification")
    print("=" * 80)

    corpus_path = "./data/veteran_rag_corpus.csv"

    try:
        df = pd.read_csv(corpus_path)
        print(f"✓ Corpus loaded: {len(df)} chunks")

        # Check for enriched sources
        enriched_sources = [
            "PTSD Clinical Definition",
            "PTSD Symptom Criteria DSM-5",
            "CPT Clinical Description",
            "Veterans Crisis Line",
        ]

        found_sources = []
        for source in enriched_sources:
            matches = df[df["source"].str.contains(source, case=False, na=False)]
            if len(matches) > 0:
                found_sources.append(source)
                print(f"  ✓ Found '{source}': {len(matches)} chunks")

        if len(found_sources) == len(enriched_sources):
            print("\n✓ SUCCESS: All enriched content sources found!")
            return True
        else:
            print(
                f"\n⚠ WARNING: Only found {len(found_sources)}/{len(enriched_sources)} enriched sources"
            )
            print("  Did you run: python enrich_ptsd_corpus.py?")
            return False

    except FileNotFoundError:
        print(f"✗ ERROR: Corpus file not found at {corpus_path}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_rag_system():
    """Test RAG system can load and retrieve"""
    print("\n" + "=" * 80)
    print("TEST 2: RAG System Loading")
    print("=" * 80)

    try:
        rag = VeteranHealthRAG(
            faiss_index_path="./models/faiss_index",
            chunks_path="./data/veteran_rag_corpus.csv",
        )

        # Load corpus
        rag.load_corpus()
        print(f"✓ Corpus loaded: {len(rag.chunks_df)} chunks")

        # Check for index
        index_path = Path("./models/faiss_index/index.faiss")
        if index_path.exists():
            rag.load_index()
            print(f"✓ FAISS index loaded: {rag.faiss_index.ntotal} vectors")
        else:
            print("⚠ No FAISS index found - will create on first query")
            rag.create_embeddings(show_progress=False)
            rag.build_faiss_index(save=True)
            print(f"✓ FAISS index created: {rag.faiss_index.ntotal} vectors")

        return rag

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_definition_query(rag):
    """Test that 'What is PTSD?' returns good results"""
    print("\n" + "=" * 80)
    print("TEST 3: Definition Query Test")
    print("=" * 80)

    query = "What is PTSD?"
    print(f"Query: {query}")
    print("-" * 80)

    try:
        # Retrieve chunks
        results = rag.retrieve(query, k=5)

        print(f"\n✓ Retrieved {len(results)} chunks")
        print("\nTop 3 Results:")
        for i, row in results.head(3).iterrows():
            print(f"\n{i + 1}. Score: {row['score']:.4f}")
            print(f"   Source: {row['source']}")
            print(f"   Text: {row['text'][:150]}...")

        # Check if we got good results
        top_result = results.iloc[0]

        # Success criteria
        has_good_score = top_result["score"] > 0.5
        has_substantive_content = len(top_result["text"]) > 150
        has_definition_terms = any(
            term in top_result["text"].lower()
            for term in [
                "mental health",
                "disorder",
                "condition",
                "characterized",
                "trauma",
            ]
        )

        print(f"\nResult Quality Check:")
        print(
            f"  Score > 0.5: {'✓' if has_good_score else '✗'} ({top_result['score']:.4f})"
        )
        print(
            f"  Substantive (>150 chars): {'✓' if has_substantive_content else '✗'} ({len(top_result['text'])} chars)"
        )
        print(f"  Contains definition terms: {'✓' if has_definition_terms else '✗'}")

        if has_good_score and has_substantive_content and has_definition_terms:
            print("\n✓ SUCCESS: Query returns high-quality definition content!")
            return True
        else:
            print("\n⚠ WARNING: Query results could be better")
            print("  Consider running enrichment again or rebuilding index")
            return False

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_answer_generation(rag):
    """Test full answer generation"""
    print("\n" + "=" * 80)
    print("TEST 4: Answer Generation Test")
    print("=" * 80)

    test_queries = [
        "What is PTSD?",
        "What are common PTSD symptoms?",
        "What treatments are available for PTSD?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)

        try:
            result = rag.answer_query(query, k=5, use_llm=False)

            print(f"Answer ({len(result['answer'])} chars):")
            print(result["answer"][:300])
            if len(result["answer"]) > 300:
                print("...")

            print(f"\nSources used: {result['num_sources']}")
            print(f"Top score: {result['top_score']:.3f}")

            # Check answer quality
            has_content = len(result["answer"]) > 100
            has_sources = result["num_sources"] > 0
            has_good_score = result["top_score"] > 0.5

            if has_content and has_sources and has_good_score:
                print("✓ Good quality answer")
            else:
                print("⚠ Answer quality could be improved")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback

            traceback.print_exc()


def test_method_availability(rag):
    """Test that new formatting methods are available"""
    print("\n" + "=" * 80)
    print("TEST 5: Method Availability Check")
    print("=" * 80)

    required_methods = [
        "_format_definition_answer",
        "_format_treatment_answer",
        "_format_standard_answer",
        "_extractive_answer",
    ]

    missing_methods = []
    for method_name in required_methods:
        if hasattr(rag, method_name):
            print(f"✓ {method_name} - Available")
        else:
            print(f"✗ {method_name} - MISSING")
            missing_methods.append(method_name)

    if not missing_methods:
        print("\n✓ SUCCESS: All required methods available!")
        return True
    else:
        print(f"\n⚠ WARNING: Missing methods: {', '.join(missing_methods)}")
        print("  Add these methods to your rag_system.py VeteranHealthRAG class")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("VETERAN MENTAL HEALTH RAG - ENRICHMENT VERIFICATION")
    print("=" * 80)

    results = {
        "corpus_enrichment": False,
        "rag_loading": False,
        "definition_query": False,
        "methods_available": False,
    }

    # Test 1: Corpus enrichment
    results["corpus_enrichment"] = test_corpus_enrichment()

    # Test 2: RAG system loading
    rag = test_rag_system()
    if rag:
        results["rag_loading"] = True

        # Test 3: Definition query
        results["definition_query"] = test_definition_query(rag)

        # Test 4: Answer generation (informational, not pass/fail)
        test_answer_generation(rag)

        # Test 5: Method availability
        results["methods_available"] = test_method_availability(rag)

    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name.replace('_', ' ').title()}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! Your chatbot is ready!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Run: streamlit run app.py")
        print("2. Try query: 'What is PTSD?'")
        print("3. Compare with previous responses")
    else:
        print("\n" + "=" * 80)
        print("⚠ SOME TESTS FAILED")
        print("=" * 80)

        if not results["corpus_enrichment"]:
            print("\n→ Run enrichment: python enrich_ptsd_corpus.py")

        if not results["methods_available"]:
            print("\n→ Add improved methods to rag_system.py")
            print("  See: rag_improvements_patch.py")

        if not results["definition_query"]:
            print("\n→ Rebuild FAISS index: rm -rf ./models/faiss_index/")
            print("  Then restart app to rebuild automatically")

    print()


if __name__ == "__main__":
    main()

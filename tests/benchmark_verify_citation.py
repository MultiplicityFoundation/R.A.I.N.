import time
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

try:
    from rain_lab_meeting_chat_version import ContextManager, Config
except ImportError as e:
    print(f"Error: Could not import rain_lab_meeting_chat_version. Ensure you are running from repo root. Details: {e}")
    sys.exit(1)

def run_benchmark():
    # Setup ContextManager with many dummy papers
    print("Setting up benchmark environment...")
    # Using dummy path since we mock loaded_papers directly
    config = Config(library_path=".")
    cm = ContextManager(config)

    # Mock loaded papers: 1000 papers, each with ~1000 words
    # We want realistic enough content so verify_citation has to scan through it
    num_papers = 1000
    base_content = "This is a dummy text that does NOT contain the target quote at all. " * 10

    # Fill loaded_papers with content that DOES NOT contain the quote (worst case scan)
    # except the last one
    for i in range(num_papers):
        paper_name = f"paper_{i}.md"
        content = f"Paper {i} content. {base_content}"
        cm.loaded_papers[paper_name] = content
        cm.loaded_papers_lower[paper_name] = content.lower()

    # The quote to verify
    quote = "sample text containing some words"

    # Add the quote to the LAST paper to force full scan (worst case)
    last_paper = f"paper_{num_papers-1}.md"
    cm.loaded_papers[last_paper] += f" {quote} "
    cm.loaded_papers_lower[last_paper] += f" {quote.lower()} "

    print(f"Running benchmark with {num_papers} papers...")
    print(f"Quote: '{quote}' (located in the last paper)")

    iterations = 2000
    start_time = time.perf_counter()

    found_count = 0
    for _ in range(iterations):
        result = cm.verify_citation(quote, fuzzy=True)
        if result:
            found_count += 1

    end_time = time.perf_counter()
    duration = end_time - start_time

    print(f"Completed {iterations} iterations.")
    print(f"Total time: {duration:.4f} seconds")
    print(f"Average time per call: {duration/iterations*1000:.4f} ms")
    print(f"Found matches: {found_count}/{iterations}")

    # Verify negative case
    print("\nVerifying negative case...")
    quote_missing = "This quote definitely does not exist in any paper whatsoever"
    result_missing = cm.verify_citation(quote_missing, fuzzy=True)
    if result_missing is None:
        print("✓ correctly returned None for missing quote")
    else:
        print(f"✗ incorrect result for missing quote: {result_missing}")

if __name__ == "__main__":
    run_benchmark()

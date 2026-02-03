import random
import os
from tqdm import tqdm

def sample_random_jsonl(input_file_path: str, output_file_path: str, sample_size: int, random_seed: int):
    """
    Randomly sample a fixed number of lines from a JSONL file and write to a new file.

    Uses a two-pass scan for memory efficiency (works well for large files).

    Args:
        input_file_path: input JSONL file path
        output_file_path: output JSONL file path
        sample_size: number of lines to sample
        random_seed: random seed for reproducibility
    """
    print("--- Starting random sampling ---")
    print(f"  [config] input:  {input_file_path}")
    print(f"  [config] output: {output_file_path}")
    print(f"  [config] sample_size: {sample_size}")
    print(f"  [config] random_seed: {random_seed}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # --- Pass 1: count total lines ---
        print("\n[step 1/3] Counting total lines...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in tqdm(f, desc="counting"))
        
        print(f"Total lines: {total_lines}")

        if sample_size > total_lines:
            print(f"Warning: sample_size ({sample_size}) > total_lines ({total_lines}); sampling all lines.")
            sample_size = total_lines

        # --- Pass 2: pick indices to sample ---
        print("\n[step 2/3] Generating random indices...")
        random.seed(random_seed)
        # Use random.sample to efficiently pick unique indices.
        indices_to_sample = random.sample(range(total_lines), k=sample_size)
        # Convert to a set for O(1) average membership checks.
        indices_to_sample_set = set(indices_to_sample)
        print(f"Generated {len(indices_to_sample_set)} unique indices.")

        # --- Pass 3: write sampled lines ---
        print("\n[step 3/3] Writing sampled lines...")
        lines_written = 0
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            # Progress bar via tqdm
            for i, line in tqdm(enumerate(infile), total=total_lines, desc="sampling"):
                if i in indices_to_sample_set:
                    outfile.write(line)
                    lines_written += 1
        
        print(f"\nWrote {lines_written} line(s) to '{output_file_path}'.")
        print("--- Sampling complete ---")

    except FileNotFoundError:
        print(f"Error: input file not found: '{input_file_path}'")
    except Exception as e:
        print(f"Unexpected error while sampling: {e}")

if __name__ == "__main__":
    INPUT_FILE = os.getenv("INPUT_FILE", "/path/to/input.jsonl")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "/path/to/output.jsonl")
    SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "2000"))
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

    # Run
    sample_random_jsonl(
        input_file_path=INPUT_FILE,
        output_file_path=OUTPUT_FILE,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED
    )

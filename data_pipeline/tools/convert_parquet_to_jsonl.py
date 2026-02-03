import os
import pandas as pd

# Fixed parameters (edit as needed)
input_dir = os.getenv("INPUT_DIR", "/path/to/parquet_dir")  # directory containing parquet files
output_file = os.getenv("OUTPUT_FILE", "/path/to/output.jsonl")  # output JSONL file

def convert_parquet_to_jsonl(input_dir, output_file):
    dfs = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".parquet"):
            file_path = os.path.join(input_dir, filename)
            print(f"Reading: {file_path}")
            df = pd.read_parquet(file_path)
            dfs.append(df)
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        all_df.to_json(output_file, orient="records", lines=True, force_ascii=False)
        print(f"Saved to: {output_file}")
    else:
        print("No parquet files found.")

if __name__ == "__main__":
    convert_parquet_to_jsonl(input_dir, output_file)

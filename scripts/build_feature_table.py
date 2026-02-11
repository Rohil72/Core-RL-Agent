
import os
import argparse
import json
import logging
import pandas as pd
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from src.data.feature_engineering import assemble_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CYCLES_DIR = os.path.join("data", "cycles")
FEATURE_TABLES_DIR = os.path.join("data", "feature_tables")
os.makedirs(FEATURE_TABLES_DIR, exist_ok=True)

def process_file(jsonl_path: str, rebuild: bool = False) -> str:
    """
    Process a single JSONL file and return status.
    """
    ticker = os.path.splitext(os.path.basename(jsonl_path))[0]
    out_path = os.path.join(FEATURE_TABLES_DIR, f"{ticker}.parquet")
    
    if os.path.exists(out_path) and not rebuild:
        logger.info(f"Skipping {ticker}, already exists.")
        return f"Skipped {ticker}"
        
    logger.info(f"Processing {ticker}...")
    
    features_list = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    cycle_example = json.loads(line)
                    # Assemble features
                    # Note: assemble_features might call fetcher, which might access disk/network.
                    # In true multiprocessing, this might cause contention on locks if fetcher uses sqlite, 
                    # but here it uses files. Reading JSON cache is safe for read. 
                    # Fetching from web in multiprocessing is risky if rate limited.
                    # Assuming data is cached or we accept limits.
                    feat = assemble_features(cycle_example)
                    features_list.append(feat)
                except Exception as e:
                    logger.error(f"Error processing line in {jsonl_path}: {e}")
                    
        if features_list:
            df = pd.DataFrame(features_list)
            # Ensure proper types
            # Convert object columns to reasonable types if possible
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass
                        
            df.to_parquet(out_path)
            return f"Processed {ticker}: {len(df)} rows"
        else:
            return f"No valid cycles for {ticker}"
            
    except Exception as e:
        logger.error(f"Failed to process {jsonl_path}: {e}")
        return f"Failed {ticker}"

def main():
    parser = argparse.ArgumentParser(description="Build feature tables from cycle examples.")
    parser.add_argument("--tickers", type=str, help="Comma separated list of tickers to process. Default all.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild existing parquet files.")
    parser.add_argument("--parallel_jobs", type=int, default=1, help="Number of parallel jobs.")
    
    args = parser.parse_args()
    
    # Identify files
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
        files = []
        for t in tickers:
            p = os.path.join(CYCLES_DIR, f"{t}.jsonl")
            if os.path.exists(p):
                files.append(p)
            else:
                logger.warning(f"File not found for {t}: {p}")
    else:
        files = glob.glob(os.path.join(CYCLES_DIR, "*.jsonl"))
        
    logger.info(f"Found {len(files)} files to process.")
    
    # Manifest
    manifest = {
        "files_processed": [],
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Process
    if args.parallel_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.parallel_jobs) as executor:
            futures = {executor.submit(process_file, f, args.rebuild): f for f in files}
            for future in as_completed(futures):
                res = future.result()
                logger.info(res)
    else:
        for f in files:
            res = process_file(f, args.rebuild)
            logger.info(res)
            
    # Update manifest
    # List all parquet files
    parquet_files = glob.glob(os.path.join(FEATURE_TABLES_DIR, "*.parquet"))
    manifest["files_count"] = len(parquet_files)
    manifest["files_list"] = [os.path.basename(p) for p in parquet_files]
    
    with open(os.path.join(FEATURE_TABLES_DIR, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()

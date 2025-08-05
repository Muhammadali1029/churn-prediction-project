"""Analyse logs to understand model training patterns"""
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

def analyse_json_logs(log_dir: Path):
    """Analyse structured JSON logs"""
    log_files = list(log_dir.glob("*_structured.json"))
    
    all_logs = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    all_logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
    
    # Convert to DataFrame for analysis
    df_logs = pd.DataFrame(all_logs)
    
    if df_logs.empty:
        print("No logs found!")
        return
    
    print("=== LOG ANALYSIS ===")
    print(f"Total log entries: {len(df_logs)}")
    print(f"\nLog levels:")
    print(df_logs['level'].value_counts())
    
    # Find all model training logs
    model_logs = df_logs[df_logs['message'].str.contains('Model', na=False)]
    if not model_logs.empty:
        print(f"\nModel training events: {len(model_logs)}")
        
    # Find any errors
    errors = df_logs[df_logs['level'] == 'ERROR']
    if not errors.empty:
        print(f"\nErrors found: {len(errors)}")
        for _, error in errors.iterrows():
            print(f"  - {error['timestamp']}: {error['message']}")
    
    return df_logs

if __name__ == "__main__":
    log_dir = Path(__file__).parent.parent / "logs"
    analyse_json_logs(log_dir)
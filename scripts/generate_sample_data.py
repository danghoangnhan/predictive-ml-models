#!/usr/bin/env python
"""Generate synthetic sample data for demonstration."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Setup paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "sample"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_health_data(n_patients: int = 50, n_records_per_patient: int = 10):
    """Generate synthetic healthcare data (GAD-7 scores)."""
    print(f"Generating {n_patients} patients with {n_records_per_patient} records each...")
    
    data = []
    for patient_id in range(1, n_patients + 1):
        for record in range(n_records_per_patient):
            timestamp = datetime.now() - timedelta(days=100 - record * 10)
            
            # GAD-7 scores tend to correlate
            base_score = np.random.randint(5, 21)
            gad7_score = max(0, min(21, base_score + np.random.randint(-3, 4)))
            
            journal_texts = [
                "Feeling anxious today",
                "Had a good day, feeling calm",
                "Worried about upcoming tasks",
                "Relaxed and peaceful",
                "Stressed about work",
            ]
            
            label = 1 if gad7_score > 15 else 0
            
            data.append({
                'patient_id': f'P{patient_id:03d}',
                'gad7_score': gad7_score,
                'journal_text': np.random.choice(journal_texts),
                'timestamp': timestamp.isoformat(),
                'label': label
            })
    
    df = pd.DataFrame(data)
    output_path = DATA_DIR / "health_scores.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}: {df.shape}")
    return df


def generate_finance_data(n_symbols: int = 5, n_records: int = 50):
    """Generate synthetic stock OHLCV data."""
    print(f"Generating {n_symbols} symbols with {n_records} records each...")
    
    data = []
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'][:n_symbols]
    
    for symbol in symbols:
        price = np.random.uniform(50, 300)
        
        for i in range(n_records):
            open_price = price + np.random.uniform(-5, 5)
            close_price = price + np.random.uniform(-5, 5)
            high_price = max(open_price, close_price) + np.random.uniform(0, 5)
            low_price = min(open_price, close_price) - np.random.uniform(0, 5)
            volume = np.random.uniform(1000000, 10000000)
            
            # Assign pattern label
            patterns = [0, 1, 2, 3]  # triangle, wedge, flag, other
            pattern = np.random.choice(patterns, p=[0.25, 0.25, 0.25, 0.25])
            
            date = datetime.now() - timedelta(days=n_records - i)
            
            data.append({
                'symbol': symbol,
                'date': date.isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(volume),
                'label': pattern
            })
            
            price = close_price
    
    df = pd.DataFrame(data)
    output_path = DATA_DIR / "stock_patterns.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}: {df.shape}")
    return df


def main():
    """Generate all sample data."""
    print("Generating synthetic sample data...\n")
    
    health_df = generate_health_data()
    print(f"Health data sample:\n{health_df.head()}\n")
    
    finance_df = generate_finance_data()
    print(f"Finance data sample:\n{finance_df.head()}\n")
    
    print("Sample data generation complete!")


if __name__ == "__main__":
    main()

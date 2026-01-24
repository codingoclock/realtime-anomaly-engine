#!/usr/bin/env python3
"""
Example: Adding transactions to the Live Transaction Stream dashboard

This example demonstrates how to populate the live transaction stream
with sample transactions for testing/demo purposes.
"""

from datetime import datetime, timedelta
import time
from dashboard.app import add_transaction_to_stream
import random


def simulate_transaction_stream(num_transactions=100, duration_seconds=60, anomaly_rate=0.1):
    """
    Simulate a stream of transactions and add them to the dashboard.
    
    Args:
        num_transactions: Total number of transactions to simulate
        duration_seconds: Time span over which to spread transactions
        anomaly_rate: Fraction of transactions that are anomalous (0.0-1.0)
    """
    print(f"Simulating {num_transactions} transactions over {duration_seconds}s...")
    
    start_time = datetime.now()
    interval = duration_seconds / num_transactions
    
    users = [f"user_{i}" for i in range(1, 21)]  # 20 different users
    
    for i in range(num_transactions):
        # Calculate timestamp
        elapsed = i * interval
        timestamp = (start_time + timedelta(seconds=elapsed)).isoformat()
        
        # Random user
        user_id = random.choice(users)
        
        # Determine if anomalous
        is_anomaly = random.random() < anomaly_rate
        
        if is_anomaly:
            # Anomalous transactions: high amount, high score
            amount = random.uniform(1000, 9999)
            anomaly_score = random.uniform(0.7, 1.0)
        else:
            # Normal transactions: low amount, low score
            amount = random.uniform(10, 500)
            anomaly_score = random.uniform(0.0, 0.3)
        
        # Add to stream
        add_transaction_to_stream(
            timestamp=timestamp,
            user_id=user_id,
            amount=amount,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly
        )
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"  Added {i + 1}/{num_transactions} transactions")
        
        # Sleep to simulate real-time flow
        time.sleep(interval)
    
    print(f"✓ Simulation complete!")


def demo_single_transactions():
    """Demo: Add individual transactions programmatically."""
    
    print("\n=== Demo: Adding Individual Transactions ===\n")
    
    # Normal transaction
    print("Adding normal transaction...")
    add_transaction_to_stream(
        timestamp=datetime.now().isoformat(),
        user_id="user_123",
        amount=250.00,
        anomaly_score=0.15,
        is_anomaly=False
    )
    print("✓ Added: user_123, $250.00, score=0.15")
    
    # Anomalous transaction
    print("\nAdding anomalous transaction...")
    add_transaction_to_stream(
        timestamp=datetime.now().isoformat(),
        user_id="user_456",
        amount=5000.00,
        anomaly_score=0.92,
        is_anomaly=True
    )
    print("✓ Added: user_456, $5000.00, score=0.92")
    
    # Extreme anomaly
    print("\nAdding extreme anomaly...")
    add_transaction_to_stream(
        timestamp=datetime.now().isoformat(),
        user_id="user_999",
        amount=25000.00,
        anomaly_score=0.99,
        is_anomaly=True
    )
    print("✓ Added: user_999, $25000.00, score=0.99")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Live Transaction Stream - Example Script")
    print("="*50)
    
    # Run demo
    demo_single_transactions()
    
    # Uncomment to run simulation:
    # simulate_transaction_stream(
    #     num_transactions=100,
    #     duration_seconds=60,
    #     anomaly_rate=0.15
    # )
    
    print("\n💡 Tip: Open the Streamlit dashboard to see these transactions in real-time!")
    print("   Command: streamlit run dashboard/app.py")

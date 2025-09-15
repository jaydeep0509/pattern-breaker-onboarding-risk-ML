import pandas as pd
import numpy as np

def simulate_onboarding_data(n=1000, seed=42):
    np.random.seed(seed)
    customer_ids = [f'CUST{i:04d}' for i in range(n)]
    agent_ids = np.random.choice([f'AGENT{j:03d}' for j in range(50)], size=n)
    base_time = pd.Timestamp('2025-07-01')
    onboarding_time = [base_time + pd.Timedelta(days=int(x)) + pd.Timedelta(hours=int(h)) 
                       for x, h in zip(np.random.uniform(0, 60, n), np.random.uniform(0, 23, n))]
    devices = np.random.choice([f'DEV{i:03d}' for i in range(20)], size=n)
    states = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Surat', 'Jaipur']
    geo_state = np.random.choice(states, size=n)
    kyc_time = np.random.exponential(scale=30, size=n) + 10
    is_normal_login_time = [1 if 8 <= t.hour < 20 else 0 for t in onboarding_time]
    fraud = np.zeros(n)
    fraud_indices = np.random.choice(n, size=int(0.05 * n), replace=False)
    fraud[fraud_indices] = 1
    for idx in fraud_indices:
        onboarding_time[idx] = base_time + pd.Timedelta(days=np.random.uniform(0, 60)) + pd.Timedelta(hours=np.random.choice([0,1,2,3,4,22,23]))
        devices[idx] = np.random.choice(devices)
        geo_state[idx] = np.random.choice(states)
        kyc_time[idx] = np.random.exponential(scale=60) + 40
        is_normal_login_time[idx] = 0
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'agent_id': agent_ids,
        'onboarding_time': onboarding_time,
        'device_id': devices,
        'geo_state': geo_state,
        'kyc_time_mins': kyc_time,
        'is_normal_login_time': is_normal_login_time,
        'fraud_label': fraud.astype(int)
    })
    return df

if __name__ == '__main__':
    df = simulate_onboarding_data()
    df.to_csv('../data/onboarding_simulated.csv', index=False)
    print("Simulated onboarding dataset created and saved.")

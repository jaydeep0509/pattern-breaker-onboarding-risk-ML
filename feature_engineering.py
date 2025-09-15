
import pandas as pd
import numpy as np

def load_data(path='../data/onboarding_simulated.csv'):
    """Load the simulated onboarding data."""
    return pd.read_csv(path, parse_dates=['onboarding_time'])

def engineer_features(df):
    # 1. Time-based Features
    df['day_of_week'] = df['onboarding_time'].dt.dayofweek  # Monday=0, Sunday=6
    df['hour_of_day'] = df['onboarding_time'].dt.hour

    # 2. Agent-Customer matching (flag weird geo-agent combos)
    df['agent_location_match'] = np.where(df['agent_id'].str[-1] == df['geo_state'].str[0], 1, 0)

    # 3. Device consistency (simulate device change for each agent)
    agent_dev_counts = df.groupby('agent_id')['device_id'].nunique()
    df['agent_device_variety'] = df['agent_id'].map(agent_dev_counts)
    df['device_flag'] = (df['agent_device_variety'] > 3).astype(int)  # Tunable rule

    # 4. KYC time risk
    df['kyc_quick'] = (df['kyc_time_mins'] < 15).astype(int)
    df['kyc_slow'] = (df['kyc_time_mins'] > 90).astype(int)

    # 5. Night login flag
    df['night_login'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)

    return df

if __name__ == "__main__":
    df = load_data()
    df_feat = engineer_features(df)
    df_feat.to_csv('../data/onboarding_features.csv', index=False)
    print("Feature engineered data saved to ../data/onboarding_features.csv")

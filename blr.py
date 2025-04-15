import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import arviz as az
from datetime import datetime, timedelta

# ----- 1. DATA PREPARATION -----

# Let's assume we have historical daily data like this:
# Create a sample dataset spanning several months
def create_sample_data(start_date='2024-01-01', end_date='2025-03-31'):
    """Create sample direct debit data with day-of-week effects and holiday impacts"""
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Create base dataframe with dates
    df = pd.DataFrame({
        'date': dates,
        'day_of_week': dates.dayofweek,  # 0=Monday, 6=Sunday
        'day_of_month': dates.day,
        # This is a placeholder - in real implementation you'd use a holiday calendar
        'is_holiday': np.random.choice([0, 1], size=len(dates), p=[0.97, 0.03])  
    })
    
    # Number of debit attempts varies by day of month (higher at month start/end)
    df['num_attempts'] = 100 + 50 * np.sin(np.pi * df['day_of_month'] / 31) + np.random.normal(0, 15, len(df))
    df['num_attempts'] = df['num_attempts'].astype(int).clip(lower=50)
    
    # Base success probability
    base_prob = 0.85
    
    # Day of week effect (Friday is best day)
    dow_effect = {0: -0.02, 1: -0.01, 2: 0, 3: 0.01, 4: 0.05, 5: -0.03, 6: -0.1}
    
    # Holiday effect (negative)
    holiday_effect = -0.15
    
    # Day of month effect (worse at month end)
    df['dom_effect'] = -0.03 * np.abs(df['day_of_month'] - 15) / 15
    
    # Calculate success probability
    df['success_prob'] = base_prob
    df['success_prob'] += df['day_of_week'].map(dow_effect)
    df['success_prob'] += df['is_holiday'] * holiday_effect
    df['success_prob'] += df['dom_effect']
    df['success_prob'] = df['success_prob'].clip(0.5, 0.99)
    
    # Generate number of successes based on probability
    df['num_successes'] = np.random.binomial(df['num_attempts'], df['success_prob'])
    
    # Calculate observed success rate
    df['success_rate'] = df['num_successes'] / df['num_attempts']
    
    return df

# Create or load your data
# In a real scenario, replace this with loading your actual data
df = create_sample_data()

# ----- 2. FEATURE ENGINEERING -----

# Extract features for model
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_friday'] = (df['day_of_week'] == 4).astype(int)
df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
df['is_month_end'] = (df['day_of_month'] >= 26).astype(int)

# Optional: Add cyclical encoding for day of month and day of week
df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Split data into training and future periods
# Assume we're predicting for April 2025
cutoff_date = '2025-03-31'
train_df = df[df['date'] <= cutoff_date].copy()
print(f"Training data shape: {train_df.shape}")

# Create future dates dataframe for next month
next_month_start = pd.Timestamp(cutoff_date) + pd.Timedelta(days=1)
next_month_end = pd.Timestamp(f"{next_month_start.year}-{next_month_start.month+1}-01") - pd.Timedelta(days=1)
if next_month_start.month == 12:
    next_month_end = pd.Timestamp(f"{next_month_start.year+1}-01-01") - pd.Timedelta(days=1)

future_dates = pd.date_range(start=next_month_start, end=next_month_end)
future_df = pd.DataFrame({'date': future_dates})
future_df['day_of_week'] = future_df['date'].dt.dayofweek
future_df['day_of_month'] = future_df['date'].dt.day
future_df['month'] = future_df['date'].dt.month
future_df['year'] = future_df['date'].dt.year
future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)
future_df['is_friday'] = (future_df['day_of_week'] == 4).astype(int)
future_df['is_month_start'] = (future_df['day_of_month'] <= 5).astype(int)
future_df['is_month_end'] = (future_df['day_of_month'] >= 26).astype(int)
future_df['day_of_month_sin'] = np.sin(2 * np.pi * future_df['day_of_month'] / 31)
future_df['day_of_month_cos'] = np.cos(2 * np.pi * future_df['day_of_month'] / 31)
future_df['day_of_week_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
future_df['day_of_week_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)

# Set holidays in future data (in real implementation, use an actual holiday calendar)
future_df['is_holiday'] = np.random.choice([0, 1], size=len(future_df), p=[0.97, 0.03])

# ----- 3. PREPARE MODEL INPUTS -----

# Select features for model
features = [
    'is_friday', 'is_weekend', 'is_month_start', 'is_month_end', 
    'day_of_month_sin', 'day_of_month_cos', 'day_of_week_sin', 'day_of_week_cos',
    'is_holiday'
]

# Prepare X, successes, and attempts
X_train = train_df[features].values
successes = train_df['num_successes'].values
attempts = train_df['num_attempts'].values

# Get future features
X_future = future_df[features].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_future_scaled = scaler.transform(X_future)

# ----- 4. BUILD AND TRAIN BAYESIAN MODEL -----

with pm.Model() as model:
    # Priors for coefficients
    intercept = pm.Normal('intercept', mu=0, sigma=5)
    coeffs = pm.Normal('coeffs', mu=0, sigma=1, shape=X_train_scaled.shape[1])
    
    # Add extra variance parameter to account for overdispersion
    sigma = pm.HalfNormal('sigma', sigma=0.1)
    
    # Linear predictor
    logit_p = intercept + pm.math.dot(X_train_scaled, coeffs)
    
    # Apply sigmoid to get probabilities
    p = pm.Deterministic('p', pm.math.invlogit(logit_p))
    
    # Beta-binomial likelihood (handles overdispersion better than simple binomial)
    alpha = pm.Deterministic('alpha', p * (1/sigma**2 - 1))
    beta = pm.Deterministic('beta', (1-p) * (1/sigma**2 - 1))
    
    # Use Beta-Binomial for overdispersed binomial data
    likelihood = pm.BetaBinomial('likelihood', alpha=alpha, beta=beta, 
                                n=attempts, observed=successes)
    
    # Sample from posterior
    trace = pm.sample(1000, tune=1000, return_inferencedata=True, 
                      target_accept=0.9, cores=2)

# ----- 5. MODEL EVALUATION -----

# Posterior predictive checks
with model:
    ppc = pm.sample_posterior_predictive(trace, var_names=['p'])

# Print model summary
summary = az.summary(trace, var_names=['intercept', 'coeffs', 'sigma'])
print(summary)

# Plot feature importance (absolute values of coefficients)
feature_importance = np.abs(summary.loc[['coeffs[0]', 'coeffs[1]', 'coeffs[2]', 
                                        'coeffs[3]', 'coeffs[4]', 'coeffs[5]',
                                        'coeffs[6]', 'coeffs[7]', 'coeffs[8]'], 'mean'].values)
plt.figure(figsize=(12, 6))
plt.bar(features, feature_importance)
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----- 6. PREDICT NEXT MONTH -----

# Get estimated number of attempts for future dates
# In real implementation, use your actual forecast for number of attempts
future_df['num_attempts'] = 100 + 50 * np.sin(np.pi * future_df['day_of_month'] / 31)
future_df['num_attempts'] = future_df['num_attempts'].astype(int)

# Make predictions for future dates
with model:
    # Linear predictor for future data
    logit_p_future = intercept + pm.math.dot(X_future_scaled, coeffs)
    p_future = pm.Deterministic('p_future', pm.math.invlogit(logit_p_future))
    
    # Generate predictions
    future_preds = pm.sample_posterior_predictive(trace, 
                                                 var_names=['p_future'], 
                                                 samples=1000)

# Extract predictions
future_probs = future_preds['p_future']

# Calculate statistics for each day
future_df['pred_success_rate_mean'] = future_probs.mean(axis=0)
future_df['pred_success_rate_lower'] = np.percentile(future_probs, 2.5, axis=0)
future_df['pred_success_rate_upper'] = np.percentile(future_probs, 97.5, axis=0)

# Calculate expected successes for each day
future_df['exp_successes_mean'] = future_df['pred_success_rate_mean'] * future_df['num_attempts']
future_df['exp_successes_lower'] = future_df['pred_success_rate_lower'] * future_df['num_attempts']
future_df['exp_successes_upper'] = future_df['pred_success_rate_upper'] * future_df['num_attempts']

# ----- 7. MONTHLY AGGREGATION -----

# Calculate monthly success rate (weighted by number of attempts)
total_attempts = future_df['num_attempts'].sum()
monthly_success_rate_mean = (future_df['exp_successes_mean'].sum() / total_attempts)

# For uncertainty, we need to do this at the sample level
monthly_samples = []
n_samples = future_probs.shape[0]

for i in range(n_samples):
    # For each sample, calculate successes for each day
    daily_successes = future_probs[i] * future_df['num_attempts'].values
    # Sum up successes and divide by total attempts for monthly rate
    monthly_rate = daily_successes.sum() / total_attempts
    monthly_samples.append(monthly_rate)

# Calculate statistics for monthly success rate
monthly_success_rate_lower = np.percentile(monthly_samples, 2.5)
monthly_success_rate_upper = np.percentile(monthly_samples, 97.5)

# ----- 8. VISUALIZE RESULTS -----

# Plot daily predicted success rates with confidence intervals
plt.figure(figsize=(14, 7))
plt.plot(future_df['date'], future_df['pred_success_rate_mean'], 'b-', label='Mean prediction')
plt.fill_between(future_df['date'], 
                future_df['pred_success_rate_lower'], 
                future_df['pred_success_rate_upper'],
                alpha=0.3, color='blue', label='95% credible interval')

# Add horizontal line for monthly average
plt.axhline(y=monthly_success_rate_mean, color='r', linestyle='--', 
           label=f'Monthly avg: {monthly_success_rate_mean:.2%}')

# Format plot
plt.title('Predicted Direct Debit Success Rates for Next Month')
plt.xlabel('Date')
plt.ylabel('Success Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----- 9. REPORTING -----

print("\n---- NEXT MONTH FORECAST SUMMARY ----")
print(f"Forecast period: {future_df['date'].min().strftime('%Y-%m-%d')} to {future_df['date'].max().strftime('%Y-%m-%d')}")
print(f"Total expected attempts: {total_attempts:,}")
print(f"Expected monthly success rate: {monthly_success_rate_mean:.2%} (95% CI: {monthly_success_rate_lower:.2%} - {monthly_success_rate_upper:.2%})")
print(f"Expected total successes: {future_df['exp_successes_mean'].sum():.0f} (95% CI: {future_df['exp_successes_lower'].sum():.0f} - {future_df['exp_successes_upper'].sum():.0f})")
print("\nDay with highest expected success rate:", future_df.loc[future_df['pred_success_rate_mean'].idxmax(), 'date'].strftime('%Y-%m-%d'), 
      f"({future_df['pred_success_rate_mean'].max():.2%})")
print("Day with lowest expected success rate:", future_df.loc[future_df['pred_success_rate_mean'].idxmin(), 'date'].strftime('%Y-%m-%d'), 
      f"({future_df['pred_success_rate_mean'].min():.2%})")
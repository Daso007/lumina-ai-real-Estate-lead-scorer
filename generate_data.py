import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker to create realistic fake data
fake = Faker('en_IN')

# --- Configuration ---
NUM_USERS = 1000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 8, 31)

print("Starting data generation process...")

# --- Part 1: Generate Google Analytics Style Web Session Data ---
print("Step 1/4: Generating Google Analytics web session data...")
ga_records = []
user_ids = [fake.uuid4() for _ in range(NUM_USERS)]

for user_id in user_ids:
    num_sessions = random.randint(1, 15)  # Each user can have multiple sessions
    is_new_user = True
    for _ in range(num_sessions):
        session_date = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
        
        # Simulate user behavior patterns
        high_intent = random.random() > 0.7  # 30% of sessions are high-intent
        
        traffic_source = random.choice(['google_organic', 'facebook_cpc', 'direct', 'instagram_paid', 'referral_99acres'])
        avg_session_duration = random.uniform(30, 600) if high_intent else random.uniform(5, 120)
        engagement_rate = random.uniform(0.6, 0.95) if high_intent else random.uniform(0.1, 0.5)
        pages_viewed = random.randint(4, 15) if high_intent else random.randint(1, 4)
        viewed_villa_page = True if high_intent and random.random() > 0.5 else False
        used_emi_calculator = True if high_intent and random.random() > 0.6 else False
        
        ga_records.append({
            'user_id': user_id,
            'session_id': fake.uuid4(),
            'session_date': session_date,
            'is_new_user': is_new_user,
            'traffic_source': traffic_source,
            'avg_session_duration_seconds': round(avg_session_duration, 2),
            'engagement_rate': round(engagement_rate, 4),
            'pages_viewed': pages_viewed,
            'viewed_villa_page': viewed_villa_page,
            'used_emi_calculator': used_emi_calculator,
        })
        if is_new_user:
            is_new_user = False

ga_df = pd.DataFrame(ga_records)
ga_df.to_csv('google_analytics_data.csv', index=False)
print(f"Successfully generated {len(ga_df)} GA records into google_analytics_data.csv")


# --- Part 2: Generate CRM Data (Lead Submissions) ---
print("Step 2/4: Generating CRM lead data...")
crm_records = []
# A subset of users will submit the contact form
form_submitters = random.sample(user_ids, k=int(NUM_USERS * 0.4)) # 40% of users submit a form

for user_id in form_submitters:
    contact_date = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
    lead_source = random.choice(['Website Form', 'Zillow', 'MagicBricks'])
    
    # Let's correlate form submission with high-intent web behavior
    user_sessions = ga_df[ga_df['user_id'] == user_id]
    
    # Base probability of scheduling a visit
    scheduled_visit_prob = 0.2
    
    # Increase probability based on GA behavior
    if not user_sessions.empty:
        if user_sessions['used_emi_calculator'].any():
            scheduled_visit_prob += 0.4
        if user_sessions['viewed_villa_page'].any():
            scheduled_visit_prob += 0.25
        if user_sessions['avg_session_duration_seconds'].mean() > 240:
             scheduled_visit_prob += 0.15
            
    # Final decision to schedule a visit based on the cumulative probability
    scheduled_visit = 1 if random.random() < scheduled_visit_prob else 0

    crm_records.append({
        'user_id': user_id,
        'lead_name': fake.name(),
        'lead_email': fake.email(),
        'lead_phone': fake.phone_number(),
        'contact_date': contact_date,
        'lead_source': lead_source,
        'scheduled_site_visit': scheduled_visit # This is our target variable! 1 for Yes, 0 for No
    })

crm_df = pd.DataFrame(crm_records)
crm_df.to_csv('crm_data.csv', index=False)
print(f"Successfully generated {len(crm_df)} CRM records into crm_data.csv")

# --- Part 3: Combine the data into a single master training file ---
print("Step 3/4: Combining GA and CRM data for model training...")

# Aggregate GA data to get one row per user
user_agg_df = ga_df.groupby('user_id').agg(
    total_sessions=('session_id', 'count'),
    avg_session_duration=('avg_session_duration_seconds', 'mean'),
    avg_engagement_rate=('engagement_rate', 'mean'),
    total_pages_viewed=('pages_viewed', 'sum'),
    has_viewed_villa=('viewed_villa_page', 'max'), # 'max' on boolean gives True if any session was True
    has_used_calculator=('used_emi_calculator', 'max'),
    first_session_source=('traffic_source', 'first')
).reset_index()

# Merge the aggregated GA data with the CRM data
# We'll use a left merge to keep all CRM leads, even if some have no GA data
master_df = pd.merge(crm_df, user_agg_df, on='user_id', how='left')

master_df.to_csv('master_training_data.csv', index=False)
print(f"Successfully created master_training_data.csv with {len(master_df)} records.")

print("Step 4/4: Data generation complete!")
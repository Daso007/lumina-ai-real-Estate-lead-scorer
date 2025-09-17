import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Page Configuration ---
# Set the page title and a supportive icon
st.set_page_config(page_title="Lumina AI - Lead Scorer", page_icon="💡", layout="wide")


# --- 2. Load Your Assets (Model & Data) ---
# This function caches the loaded model and data to speed up the app
@st.cache_data
def load_assets():
    """Loads the pre-trained model and the new lead data."""
    try:
        model = joblib.load('lead_scorer_model.pkl')
        # We use crm_data as the source of "new leads" to be scored
        new_leads_df = pd.read_csv('crm_data.csv')
        full_ga_data = pd.read_csv('google_analytics_data.csv')
    except FileNotFoundError:
        st.error("Model or data files not found. Please run generate_data.py and train_model.py first.")
        return None, None, None
        
    # --- Aggregate the GA data just like in the training script ---
    user_agg_df = full_ga_data.groupby('user_id').agg(
        total_sessions=('session_id', 'count'),
        avg_session_duration=('avg_session_duration_seconds', 'mean'),
        avg_engagement_rate=('engagement_rate', 'mean'),
        total_pages_viewed=('pages_viewed', 'sum'),
        has_viewed_villa=('viewed_villa_page', 'max'),
        has_used_calculator=('used_emi_calculator', 'max'),
        first_session_source=('traffic_source', 'first')
    ).reset_index()

    # Merge with new leads to get the features needed for prediction
    new_leads_with_features = pd.merge(new_leads_df, user_agg_df, on='user_id', how='left')
    
    # Fill NA values just in case (important for live data)
    for col in new_leads_with_features.columns:
        if new_leads_with_features[col].dtype == 'object':
            new_leads_with_features[col].fillna('missing', inplace=True)
        else:
            new_leads_with_features[col].fillna(0, inplace=True)
            
    return model, new_leads_df, new_leads_with_features

model, new_leads_df, new_leads_with_features = load_assets()

# --- 3. The User Interface (UI) ---

# Title and introduction
st.title("💡 Lumina AI - Predictive Lead Prioritization")
st.subheader("Welcome, Sales Manager Priya!")
st.write("""
This dashboard analyzes incoming website leads and assigns a 'Lumina Score' to predict the likelihood of them scheduling a site visit. 
Focus your efforts on the hottest leads at the top!
""")

# Check if model and data loaded successfully
if model is not None and new_leads_with_features is not None:
    
    # --- 4. Prediction Logic ---
    # Define the feature list the model was trained on
    features = [
        'total_sessions', 'avg_session_duration', 'avg_engagement_rate', 
        'total_pages_viewed', 'has_viewed_villa', 'has_used_calculator',
        'lead_source', 'first_session_source'
    ]
    
    X_predict = new_leads_with_features[features]
    
    # Use the model to predict the *probability* of conversion
    # We predict_proba to get a score, not just a yes/no
    predicted_probabilities = model.predict_proba(X_predict)[:, 1]

    # --- 5. Displaying the Dashboard ---
    # Create the Lumina Score and add it to our dataframe
    results_df = new_leads_df.copy()
    results_df['Lumina_Score'] = np.round(predicted_probabilities * 100).astype(int)
    
    # Define a simple function for 'Next Best Action'
    def get_next_action(score):
        if score > 85:
            return "🔥 Immediate Call"
        elif score > 60:
            return "✉️ Send Personalized Villa Brochure"
        elif score > 40:
            return "📈 Add to Nurture Campaign"
        else:
            return "🗑️ Monitor / Deprioritize"
            
    results_df['Next_Best_Action'] = results_df['Lumina_Score'].apply(get_next_action)
    
    # Select and reorder columns for a clean display
    display_columns = [
        'lead_name', 
        'Lumina_Score', 
        'Next_Best_Action',
        'lead_source',
        'lead_phone',
        'lead_email'
    ]
    
    # Sort the dataframe by the highest score
    sorted_df = results_df[display_columns].sort_values(by='Lumina_Score', ascending=False).reset_index(drop=True)
    
    # Display the final, beautiful dataframe
    st.dataframe(sorted_df)

    st.success(f"Successfully scored {len(sorted_df)} new leads.")
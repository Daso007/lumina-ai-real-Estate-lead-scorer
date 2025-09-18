import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import streamlit.components.v1 as components

# --- 1. Page Configuration ---
st.set_page_config(page_title="Lumina AI - Lead Scorer", page_icon="💡", layout="wide")


# --- 2. Load Your Assets ---
@st.cache_data
def load_assets():
    """Loads all necessary assets for the app."""
    try:
        model = joblib.load('lead_scorer_model.pkl')
        new_leads_df = pd.read_csv('crm_data.csv')
        full_ga_data = pd.read_csv('google_analytics_data.csv')
    except FileNotFoundError:
        st.error("Model or data files not found. Please ensure all data and model files are in the same folder.")
        return None, None, None
        
    user_agg_df = full_ga_data.groupby('user_id').agg(
        total_sessions=('session_id', 'count'),
        avg_session_duration=('avg_session_duration_seconds', 'mean'),
        avg_engagement_rate=('engagement_rate', 'mean'),
        total_pages_viewed=('pages_viewed', 'sum'),
        has_viewed_villa=('viewed_villa_page', 'max'),
        has_used_calculator=('used_emi_calculator', 'max'),
        first_session_source=('traffic_source', 'first')
    ).reset_index()

    new_leads_with_features = pd.merge(new_leads_df, user_agg_df, on='user_id', how='left')
    
    for col in new_leads_with_features.columns:
        if new_leads_with_features[col].dtype == 'object':
            new_leads_with_features[col].fillna('missing', inplace=True)
        else:
            new_leads_with_features[col].fillna(0, inplace=True)
            
    return model, new_leads_df, new_leads_with_features

model, new_leads_df, new_leads_with_features = load_assets()


# --- 3. Action Agent Functions ---
def trigger_email_agent(lead_data):
    webhook_url = "https://hook.us2.make.com/63rvb0ygib3l6znetiob2jufpu9337yt" # <-- MAKE SURE THIS IS YOUR URL
    payload = {
        "lead_name": lead_data['lead_name'], "lead_email": lead_data['lead_email'],
        "lumina_score": str(lead_data['Lumina_Score']), "property_type": "Villa"
    }
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            st.toast(f"✅ Email campaign triggered for {lead_data['lead_name']}!")
    except Exception as e:
        st.toast(f"❌ Connection error: {e}")

def generate_vapi_call_button(lead_data, index):
    vapi_assistant_id = "c2388756-391a-401f-9457-d144d8f1f3ea" 
    vapi_public_key = "6c555660-87b3-421a-b9a1-d59b899798ab"
    lead_first_name = lead_data['lead_name'].split()[0]
    html_code = f"""
        <script src="https://cdn.vapi.ai/sip.js"></script>
        <script>
            function makeCall_{index}() {{
                const vapi = new Vapi('{vapi_public_key}');
                vapi.start('{vapi_assistant_id}', {{ 
                    variables: {{ lead_name: "{lead_first_name}" }} 
                }});
            }}
        </script>
        <button onclick="makeCall_{index}()">📞 Call Now (Web)</button>
    """
    return html_code

# --- 4. Main App UI ---
st.title("💡 Lumina AI - Predictive Lead Prioritization")
st.subheader("Welcome, Sales Manager Priya!")
st.write("This dashboard analyzes incoming website leads to prioritize the hottest prospects. Use the Action Agents to engage with them instantly.")

if model is not None and new_leads_with_features is not None:
    
    features = [
        'total_sessions', 'avg_session_duration', 'avg_engagement_rate', 
        'total_pages_viewed', 'has_viewed_villa', 'has_used_calculator',
        'lead_source', 'first_session_source'
    ]
    X_predict = new_leads_with_features[features]
    predicted_probabilities = model.predict_proba(X_predict)[:, 1]

    results_df = new_leads_df.copy()
    results_df['Lumina_Score'] = np.round(predicted_probabilities * 100).astype(int)
    
    def get_next_action(score):
        if score >= 85: return "🔥 Immediate Call"
        elif score >= 70: return "✉️ Send Brochure" # Adjusted threshold for more variety
        elif score >= 40: return "📈 Add to Nurture"
        else: return "🗑️ Monitor"
            
    results_df['Next_Best_Action'] = results_df['Lumina_Score'].apply(get_next_action)
    
    display_columns = [
        'lead_name', 'Lumina_Score', 'Next_Best_Action',
        'lead_source', 'lead_phone', 'lead_email'
    ]
    sorted_df = results_df[display_columns].sort_values(by='Lumina_Score', ascending=False).reset_index(drop=True)
    
    # --- 5. Display the Original, Clean Dashboard ---
    st.subheader("Prioritized Lead List")
    st.dataframe(sorted_df)
    st.success(f"Successfully scored {len(sorted_df)} new leads.")
    st.markdown("---")
    
    # --- 6. ADD THE INTERACTIVE ACTION HUB ---
    st.subheader("⚡ Action Hub")
    st.write("Select a high-priority lead from the list above to trigger an Action Agent.")

    # We will only show the top leads in the dropdown for actionability
    actionable_leads = sorted_df[sorted_df['Lumina_Score'] >= 70]
    lead_options = actionable_leads['lead_name'].tolist()

    if lead_options:
        selected_lead_name = st.selectbox("Select a High-Priority Lead:", options=lead_options)
        
        # Get all the data for the selected lead
        selected_lead_data = actionable_leads[actionable_leads['lead_name'] == selected_lead_name].iloc[0]
        
        # Display the action for that lead
        action = selected_lead_data['Next_Best_Action']
        st.write(f"**Recommended Action:** {action}")

        # --- Display the correct button based on the action ---
        if "Call" in action:
            # We use a unique index for the function name
            unique_index = selected_lead_data.name 
            call_button_html = generate_vapi_call_button(selected_lead_data, unique_index)
            components.html(call_button_html, height=40)
        
        elif "Brochure" in action:
            if st.button("✉️ Trigger Email & Brochure Agent"):
                trigger_email_agent(selected_lead_data)
    else:
        st.write("No high-priority leads (score >= 70) to action at this time.")
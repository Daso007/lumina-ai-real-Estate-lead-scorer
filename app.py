import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import streamlit.components.v1 as components # <-- ADD THIS IMPORT

# --- 1. Page Configuration ---
st.set_page_config(page_title="Lumina AI - Lead Scorer", page_icon="💡", layout="wide")

# --- 2. Load Your Assets ---
@st.cache_data
def load_assets():
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
    """Generates the HTML/JS for an embedded Vapi web call button with a unique ID."""
    # !!! IMPORTANT: PASTE YOUR VAPI ASSISTANT ID AND PUBLIC KEY HERE !!!
    vapi_assistant_id = "c2388756-391a-401f-9457-d144d8f1f3ea" 
    vapi_public_key = "6c555660-87b3-421a-b9a1-d59b899798ab"
    
    # We use the index to create a unique function name for each button
    html_code = f"""
        <script src="https://cdn.vapi.ai/sip.js"></script>
        <script>
            function makeCall_{index}() {{
                const vapi = new Vapi('{vapi_public_key}');
                // We pass lead name as a variable to the Vapi assistant
                vapi.start('{vapi_assistant_id}', {{ 
                    variables: {{ lead_name: "{lead_data['lead_name'].split()[0]}" }} 
                }});
            }}
        </script>
        <button onclick="makeCall_{index}()">📞 Call Now (Web)</button>
    """
    return html_code

# --- 4. The User Interface (UI) ---
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
        if score > 85: return "🔥 Immediate Call"
        elif score > 60: return "✉️ Send Brochure"
        elif score > 40: return "📈 Add to Nurture"
        else: return "🗑️ Monitor"
            
    results_df['Next_Best_Action'] = results_df['Lumina_Score'].apply(get_next_action)
    sorted_df = results_df.sort_values(by='Lumina_Score', ascending=False).reset_index(drop=True)
    
    st.subheader("Prioritized Lead List")

    col_headers = st.columns([2, 1, 2, 2])
    col_headers[0].write("**Lead Name**")
    col_headers[1].write("**Lumina Score**")
    col_headers[2].write("**Next Best Action**")
    col_headers[3].write("**Trigger Action Agent**")
    st.markdown("---")

    for index, lead in sorted_df.head(10).iterrows():
        col1, col2, col3, col4 = st.columns([2, 1, 2, 2])
        
        with col1: st.write(lead['lead_name'])
        with col2: st.write(f"**{lead['Lumina_Score']}**")
        with col3: st.write(lead['Next_Best_Action'])
        with col4:
            # --- THIS IS THE NEW CONDITIONAL LOGIC ---
            action = lead['Next_Best_Action']
            
            if "Call" in action:
                call_button_html = generate_vapi_call_button(lead, index)
                components.html(call_button_html, height=40)
            elif "Brochure" in action:
                if st.button("✉️ Send Email & Brochure", key=f"email_{index}"):
                    trigger_email_agent(lead)
            else:
                # For "Nurture" or "Monitor", we can show a disabled-style placeholder
                st.write("*(No Action Agent)*")
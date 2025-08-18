"""
Streamlit dashboard for monitoring model performance
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import requests

# Page config
st.set_page_config(
    page_title="Churn Model Monitor",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("🎯 Churn Prediction Model Monitor")
st.markdown("Real-time monitoring of model performance and business metrics")

# Sidebar
st.sidebar.header("Controls")
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 60, 30)
date_range = st.sidebar.date_input(
    "Date range",
    value=(datetime.now() - timedelta(days=7), datetime.now()),
    max_value=datetime.now()
)

# Load model metadata
@st.cache_data
def load_model_info():
    """Load model information"""
    try:
        with open("models/model_production/metadata.json", 'r') as f:
            return json.load(f)
    except:
        return {"version": "Unknown", "metrics": {}}

model_info = load_model_info()

# Header metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Version", model_info.get("version", "Unknown"))
    
with col2:
    st.metric("Model AUC", f"{model_info.get('metrics', {}).get('val_auc', 0):.3f}")
    
with col3:
    # Simulate API health check
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        api_status = "🟢 Online" if response.status_code == 200 else "🔴 Offline"
    except:
        api_status = "🔴 Offline"
    st.metric("API Status", api_status)
    
with col4:
    st.metric("Active Threshold", "0.35")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "💰 Business Impact", "🔍 Predictions", "⚠️ Alerts"])

with tab1:
    st.header("Model Performance Metrics")
    
    # Simulate performance data over time
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    performance_data = pd.DataFrame({
        'date': dates,
        'daily_auc': np.random.normal(0.84, 0.02, 30),
        'precision': np.random.normal(0.75, 0.03, 30),
        'recall': np.random.normal(0.65, 0.04, 30),
        'predictions_made': np.random.randint(100, 500, 30)
    })
    
    # Performance over time
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(performance_data, x='date', y=['daily_auc', 'precision', 'recall'],
                     title="Model Metrics Over Time")
        fig.update_layout(yaxis_title="Score", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(performance_data, x='date', y='predictions_made',
                     title="Daily Prediction Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction distribution
    st.subheader("Prediction Distribution (Last 24h)")
    
    # Simulate prediction distribution
    predictions = np.random.beta(2, 5, 1000)  # Simulated probabilities
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=predictions, nbinsx=50, name="All Predictions"))
    fig.add_trace(go.Histogram(x=predictions[predictions > 0.35], nbinsx=50, 
                              name="Above Threshold", opacity=0.6))
    fig.update_layout(title="Churn Probability Distribution",
                     xaxis_title="Probability", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Business Impact Analysis")
    
    # Business metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Customers at Risk", "1,234", delta="+5.2%")
    with col2:
        st.metric("Revenue at Risk", "$86,380", delta="+$3,240")
    with col3:
        st.metric("Retention Campaign ROI", "245%", delta="+12%")
    
    # Revenue impact by segment
    segments = pd.DataFrame({
        'Segment': ['High Value', 'Medium Value', 'Low Value'],
        'Customers': [234, 567, 433],
        'Avg Revenue': [150, 70, 30],
        'Churn Rate': [0.15, 0.25, 0.35]
    })
    segments['Revenue at Risk'] = segments['Customers'] * segments['Avg Revenue'] * segments['Churn Rate']
    
    fig = px.bar(segments, x='Segment', y='Revenue at Risk',
                 title="Revenue at Risk by Customer Segment")
    st.plotly_chart(fig, use_container_width=True)
    
    # Retention effectiveness
    st.subheader("Retention Campaign Effectiveness")
    
    campaign_data = pd.DataFrame({
        'Week': ['W1', 'W2', 'W3', 'W4'],
        'Targeted': [100, 150, 120, 180],
        'Retained': [28, 48, 42, 61],
        'Cost': [1000, 1500, 1200, 1800],
        'Revenue Saved': [23520, 40320, 35280, 51240]
    })
    campaign_data['ROI'] = (campaign_data['Revenue Saved'] - campaign_data['Cost']) / campaign_data['Cost'] * 100
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(campaign_data, x='Week', y='ROI', markers=True,
                     title="Campaign ROI Trend")
        fig.update_layout(yaxis_title="ROI %")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=campaign_data['Week'], y=campaign_data['Targeted'],
                            name='Targeted', marker_color='lightblue'))
        fig.add_trace(go.Bar(x=campaign_data['Week'], y=campaign_data['Retained'],
                            name='Retained', marker_color='darkblue'))
        fig.update_layout(title="Retention Success Rate", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Recent Predictions")
    
    # Simulate recent predictions
    recent_predictions = pd.DataFrame({
        'Customer ID': [f"CUST-{i:04d}" for i in range(10)],
        'Prediction Time': [datetime.now() - timedelta(minutes=i*5) for i in range(10)],
        'Churn Probability': np.random.beta(2, 5, 10),
        'Monthly Revenue': np.random.normal(70, 30, 10),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 10, p=[0.5, 0.3, 0.2]),
        'Tenure': np.random.randint(1, 60, 10)
    })
    recent_predictions['Risk Level'] = pd.cut(recent_predictions['Churn Probability'],
                                             bins=[0, 0.2, 0.5, 0.8, 1.0],
                                             labels=['Low', 'Medium', 'High', 'Very High'])
    recent_predictions['Action'] = recent_predictions['Risk Level'].map({
        'Low': '✅ Monitor',
        'Medium': '⚡ Email Campaign',
        'High': '📞 Call Required',
        'Very High': '🚨 Urgent Retention'
    })
    
    st.dataframe(
        recent_predictions.style.background_gradient(subset=['Churn Probability'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # Download predictions
    csv = recent_predictions.to_csv(index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with tab4:
    st.header("System Alerts")
    
    # Alert types
    alerts = [
        {"time": "2 minutes ago", "type": "⚠️ Warning", "message": "Prediction latency increased to 150ms (threshold: 100ms)"},
        {"time": "1 hour ago", "type": "ℹ️ Info", "message": "Model retrained successfully with AUC 0.847"},
        {"time": "3 hours ago", "type": "⚠️ Warning", "message": "High number of predictions for Contract='Month-to-month' (possible data drift)"},
        {"time": "1 day ago", "type": "✅ Success", "message": "Daily model validation passed. Performance within expected range."}
    ]
    
    for alert in alerts:
        st.write(f"**{alert['time']}** - {alert['type']}: {alert['message']}")
    
    # Data drift detection
    st.subheader("Data Drift Monitoring")
    
    features = ['tenure', 'MonthlyCharges', 'Contract', 'TotalCharges']
    drift_scores = np.random.uniform(0, 0.3, len(features))
    
    drift_df = pd.DataFrame({
        'Feature': features,
        'Drift Score': drift_scores,
        'Status': ['🟢 OK' if score < 0.2 else '🟡 Warning' for score in drift_scores]
    })
    
    fig = px.bar(drift_df, x='Feature', y='Drift Score', color='Drift Score',
                 color_continuous_scale='RdYlGn_r',
                 title="Feature Drift Scores (KS Statistic)")
    fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                  annotation_text="Drift Threshold")
    st.plotly_chart(fig, use_container_width=True)

# Auto-refresh
if st.button("🔄 Refresh Now"):
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")
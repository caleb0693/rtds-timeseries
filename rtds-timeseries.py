import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, time
import plotly.express as px
import statsmodels.api as sm


st.title("RTDS PROJECT")
st.text( "Caleb Ginorio, Steve Jahn, Spencer Pizzani, sonso")


# Set a seed for reproducibility
random.seed(42)
np.random.seed(42)

# Sample size
num_entries = 1000

# Generate dummy data
data = {
    "Timestamp": [datetime.now() - timedelta(minutes=15 * i) for i in range(num_entries)],
    "Detector1": np.random.rand(num_entries) * 50,  # Assuming some concentration level
    "Detector2": np.random.rand(num_entries) * 50,
    "Detector3": np.random.rand(num_entries) * 50,
    "Detector4": np.random.rand(num_entries) * 50,
    "Detector5": np.random.rand(num_entries) * 50,
    "Detector6": np.random.rand(num_entries) * 50,
    "Detector7": np.random.rand(num_entries) * 50,
    "Detector8": np.random.rand(num_entries) * 50,
    "Detector9": np.random.rand(num_entries) * 50,
    "Detector10": np.random.rand(num_entries) * 50
}

# DataFrame
df = pd.DataFrame(data)

df.head()

st.subheader("Data")

st.dataframe(df)

# Streamlit app

def main():
    st.title("Detector Concentration")

    # User inputs for limits
    twa_limit = st.sidebar.number_input("Enter TWA Limit", value=0.0)

    # Date range picker
    min_date = df["Timestamp"].dt.date.min()
    max_date = df["Timestamp"].dt.date.max()
    start_date = st.sidebar.date_input("Start Date", min_date, min_date, max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_date, max_date)

    # Time range picker
    time_options = [time(hour, minute) for hour in range(24) for minute in range(0, 60, 15)]
    start_time = st.sidebar.selectbox("Start Time", time_options, index=0)
    end_time = st.sidebar.selectbox("End Time", time_options, index=len(time_options) - 1)

    # Combine date and time
    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)

    # Detector selection with a "Select All" option
    all_detectors = [f"Detector{i}" for i in range(1, 11)]
    select_all = st.sidebar.checkbox("Select All Detectors")
    if select_all:
        selected_detectors = st.sidebar.multiselect("Select Detectors", all_detectors, default=all_detectors)
    else:
        selected_detectors = st.sidebar.multiselect("Select Detectors", all_detectors)

    # Filter the data
    mask = (df['Timestamp'] >= start_datetime) & (df['Timestamp'] <= end_datetime)
    filtered_df = df.loc[mask, ["Timestamp"] + selected_detectors]

    # Average concentration calculation and plot
    if not filtered_df.empty and selected_detectors:
        filtered_df['AvgConcentration'] = filtered_df[selected_detectors].mean(axis=1)
        fig = px.line(filtered_df, x='Timestamp', y='AvgConcentration', title='Average Concentration Over Time')
        st.plotly_chart(fig)

    if not filtered_df.empty and selected_detectors:
        filtered_df['AvgConcentration'] = filtered_df[selected_detectors].mean(axis=1)
        filtered_df['CumulativeConcentration'] = filtered_df['AvgConcentration'].cumsum()
        total_intervals = np.arange(1, len(filtered_df) + 1)
        filtered_df['TWA'] = filtered_df['CumulativeConcentration'] / total_intervals


        # Plotting the Rolling TWA graph
        fig_twa = px.line(filtered_df, x='Timestamp', y='TWA', title='Rolling Time Weighted Average')
        fig_twa.add_hline(y=twa_limit, line_dash="dash", line_color="blue", annotation_text="TWA Limit")
        st.plotly_chart(fig_twa)

            # ARIMA Prediction Section
        st.subheader("ARIMA Model Prediction")

        # Only proceed if there's enough data
        if len(filtered_df) > 0:
            # Prepare data for ARIMA model
            ts = filtered_df.set_index('Timestamp')['AvgConcentration']

            # Define and fit ARIMA model
            arima_order = (1, 1, 1)  # Modify these parameters as needed
            arima_model = sm.tsa.ARIMA(ts, order=arima_order).fit()

            # Number of future points to predict
            n_periods = st.sidebar.number_input('Number of Future Data Points to Predict', min_value=1, value=5)

            # Making predictions
            forecast_result = arima_model.get_forecast(steps=n_periods)
            future_forecast = forecast_result.predicted_mean
            last_timestamp = filtered_df['Timestamp'].iloc[-1]
            future_dates = [end_datetime + timedelta(minutes=15 * i) for i in range(1, n_periods + 1)]

            # Plot future predictions
            fig_forecast = px.line(x=future_dates, y=future_forecast, title='Future Concentration Predictions')
            st.plotly_chart(fig_forecast)

            # Check predicted values against the TWA limit
            if any(future_forecast > twa_limit):
                st.warning("Warning: Future predicted concentrations exceed the TWA limit.")
            else:
                st.success("Predicted TWA will not exceed the limit.")

        else:
            st.error("Insufficient data for ARIMA predictions.")
    else:
        st.error("Please select detectors to view data and predictions.")


if __name__ == "__main__":
    main()




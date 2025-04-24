import pandas as pd
import streamlit as st
import gdown

@st.cache_data
def load_data_from_google_drive():
    # Google Drive file ID (from your shared link)
    file_id = "1vpyfv4fMzjuGCy7H2_YsUOG1Bbq1wkuRcMDrsjpsl_E"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "data.csv"
    
    # Download the file
    gdown.download(url, output_path, quiet=False)
    
    # Load the data into a DataFrame
    return pd.read_csv(output_path)

# Load the data
data = load_data_from_google_drive()

# Display the first few rows of the dataframe to confirm it loaded correctly
st.write(data.head())

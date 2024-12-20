import os
import pickle
from io import StringIO
import pandas as pd
import streamlit as st
import sys
# sys.path.append('/opt/miniconda3/envs/dataScience/lib/python3.9/site-packages') 
from dotenv import load_dotenv
from utils.jagath import Clean,ScoreDistribution
from utils.b2 import B2
from io import BytesIO
import io
import zipfile
from utils.jagath import PCA_PAIRWISE, FilteredData
# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------

BB_NAME = 'apartments_for_rent_classified_10K.csv'

# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
load_dotenv()
# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])
 
 
# ------------------------------------------------------
#                        CACHING
# ------------------------------------------------------
@st.cache_data
def get_data(NAME):
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df_apartments = b2.get_df(NAME)
    return df_apartments


# ------------------------------------------------------
#                         APP BACKBLAZE
# ------------------------------------------------------

df_apartments = get_data(BB_NAME)



# ------------------------------
# PART 1 : Filter Data
# ------------------------------

df_cleaned1 = Clean(df_apartments)
 
# ------------------------------
# Layout for Filters: State and Bedrooms , Price and Bathrooms
# ------------------------------
 
# Get unique states and max bedrooms
states = df_cleaned1['state'].unique()
max_bedrooms = int(df_cleaned1['bedrooms'].max())
max_bathrooms = int(df_cleaned1['bathrooms'].max())
min_price = int(df_cleaned1['price'].min())
max_price = int(df_cleaned1['price'].max())

# Display filters in one row
col1, col2 = st.columns(2)
with col1:
    selected_state = st.selectbox("Select a state:", sorted(states))
with col2:
    selected_bedrooms = st.selectbox(f"Select bedrooms (1 to {max_bedrooms}):", range(1, max_bedrooms + 1))

col3,col4 = st.columns(2)
with col3:
    selected_bathrooms = st.selectbox(f'Select bathrooms (1 to {max_bathrooms}):',range(1, max_bathrooms+1))
with col4:
    selected_price = st.slider(
    "Select price range:",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=50
)
#commented out as score based moved to new tab
#col5,col6 = st.columns(2)
#with col5:
 #   bedroom_rate = st.selectbox('Bedroom rating',range(10,0,-1))
#with col6:
 #   bathroom_rate = st.selectbox('Bathroom rating',range(10,0,-1))


# ------------------------------
# Create st.session to hold apartments after user selects options
# ------------------------------
# Initialize session state for filtered data and recommended apartments
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = pd.DataFrame()
if "recommended_data" not in st.session_state:
    st.session_state.recommended_data = pd.DataFrame()
filtered_data = []
if "data" not in st.session_state:
    st.session_state.data = []
# Button to apply filters
if st.button("Show Filtered Apartments"):
    # Filter the data based on selections
    filtered_data = df_cleaned1[
    (df_cleaned1['state'] == selected_state) &
    (df_cleaned1['bedrooms'] == selected_bedrooms)&
    (df_cleaned1['price']>=selected_price[0]) &
    (df_cleaned1['price']<= selected_price[1])
    &(df_cleaned1['bathrooms']== selected_bathrooms)]
    chunk_size = 5
    # Split the DataFrame into chunks of 5 rows each
    df_chunks = [
        filtered_data.iloc[i:i + chunk_size]
        for i in range(0, len(filtered_data), chunk_size)
    ]
    # Store the list of DataFrames in session state
    st.session_state.data = df_chunks  # Overwrite with the new chunks
    
#-------------------------------------------------------------------------
#               SHOW APARTMENTS FROM FILTERED
#-------------------------------------------------------------------------
def display_apartments(data):

    for _, row in data.iterrows():
        st.markdown("---")  # Horizontal line to separate listings
        st.markdown(f"## 🏢 {row['title']}")  # Title of the apartment
        
        with st.container():
            col1, col2 = st.columns([2, 1])  # Adjusting column width ratios for better layout
            
            # Left column details
            with col1:
                st.markdown(f"**Description:** {row['body']}")
                st.markdown(f"**Square Feet:** {row['square_feet']} sqft")
                st.markdown(f"**Bedrooms:** {row['bedrooms']} 🛏️")
                st.markdown(f"**Bathrooms:** {row['bathrooms']} 🛁")
                st.markdown(f"**Half Bathrooms:** {row['half_bathrooms']}")
                st.markdown(f"**Price:** ${row['price']:,.2f}")
                st.markdown(f"**Price Type:** {row['price_type']} ({row['currency']})")
                st.markdown(f"**Fee:** ${row['fee']}")
                st.markdown(f"**Address:** {row['address']}, {row['cityname']}, {row['state']}")
                st.markdown(f"**Source:** {row['source']}")
                st.markdown(f"**Time Listed:** {row['time']}")
            
            # Right column details
            with col2:
                st.markdown(f"**Amenities:** {row['amenities']}")
                st.markdown(f"**Pets Allowed:** {'Yes' if row['pets_allowed'] else 'No'}")
                st.markdown(f"**Cats Allowed:** {'Yes' if row['cats_allowed'] else 'No'}")
                st.markdown(f"**Dogs Allowed:** {'Yes' if row['dogs_allowed'] else 'No'}")
                st.markdown(f"**Has Photo:** {'Yes' if row['has_photo'] else 'No'}")
                st.markdown(f"**Latitude:** {row['latitude']}")
                st.markdown(f"**Longitude:** {row['longitude']}")
if "counter" not in st.session_state:
    st.session_state.counter = 0 

# Function to increment the counter
def increment_counter():
    if (st.session_state.counter < (len(st.session_state.data) - 1)):
        st.session_state.counter += 1  # Update session state variable

# Function to decrement the counter
def decrement_counter():
    if st.session_state.counter > 0:  # Prevent negative values
        st.session_state.counter -= 1

if len(st.session_state.data) > 0:
    # Layout for the buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            decrement_counter()
    with col2:
        if st.button("Next"):
            increment_counter()
            
# Check if the session state data has chunks and the counter is within range
if "data" in st.session_state and len(st.session_state.data) > 0:
    display_apartments(st.session_state.data[st.session_state.counter])

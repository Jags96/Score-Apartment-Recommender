import os
import pickle
from io import StringIO
import pandas as pd
import streamlit as st
import sys
from utils.jagath import Clean, ScoreDistribution
from utils.b2 import B2
from io import BytesIO
import io
import zipfile
from utils.jagath import PCA_PAIRWISE, FilteredData
from app import df_cleaned1, display_apartments



# Get unique states and max bedrooms
states = df_cleaned1['state'].unique()
max_bedrooms = int(df_cleaned1['bedrooms'].max())
max_bathrooms = int(df_cleaned1['bathrooms'].max())
min_price = int(df_cleaned1['price'].min())
max_price = int(df_cleaned1['price'].max())

# Display filters in one row
col1, col2 = st.columns(2)
with col1:
    selected_state = st.selectbox("Select a state:", sorted(states), key="state_selectbox")
with col2:
    selected_bedrooms = st.selectbox(f"Select bedrooms (1 to {max_bedrooms}):", range(1, max_bedrooms + 1), key="bed_selectbox" )

col3,col4 = st.columns(2)
with col3:
    selected_bathrooms = st.selectbox(f'Select bathrooms (1 to {max_bathrooms}):',range(1, max_bathrooms+1), key="bath_selectbox")
with col4:
    selected_price = st.slider(
    "Select price range:",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price), key="slide_selectbox",
    step=50
)
col5,col6 = st.columns(2)
with col5:
    bedroom_rate = st.selectbox('Bedroom rating',range(10,0,-1))
with col6:
    bathroom_rate = st.selectbox('Bathroom rating',range(10,0,-1))



# ------------------------------
# Create st.session to hold apartments after user selects options
# ------------------------------

# Button to apply filters
if st.button("Show Filtered Apartments"):
    # Filter the data based on selections
    filtered_data = df_cleaned1[
    (df_cleaned1['state'] == selected_state) &
    # (df_cleaned1['bedrooms'] == selected_bedrooms)&
    (df_cleaned1['price']>=selected_price[0]) &
    (df_cleaned1['price']<= selected_price[1])
    # &(df_cleaned1['bathrooms']== selected_bathrooms)
    ]






#-------------------------------------------------------------------------
#               GET SCORE BASED TOP APARTMENTS
#-------------------------------------------------------------------------

    st.write("Score Based Top Apartments")
    # Assuming `df_filtered` is already available and contains all the columns provided
    # Ensure `df_filtered` is a pandas DataFrame
    #used current session df Jagath may change back to filtered_data
    df_filtered = pd.DataFrame(filtered_data)
    try:
        ### Adding the scoring system/methods
        bathroom_dis = ScoreDistribution(df_filtered['bathrooms'], selected_bathrooms, bathroom_rate)
        bedroom_dis = ScoreDistribution(df_filtered['bedrooms'], selected_bedrooms, bedroom_rate)
    except:
        st.error("No data available to display.")
        st.stop()

    np_score = bedroom_dis.apply_score() + bathroom_dis.apply_score()
    df_filtered['score'] = np_score
    df_filtered = df_filtered.sort_values(by = 'score', ascending= False)
    #### df_cleaned1 is sorted based on the score
    if df_cleaned1.empty:
        st.error("No data available to display.")
        st.stop()  # Stop the script execution if no data
    # Function to display apartments in a custom card-like format
   
    # Pagination controls
    rows_per_page = 5  # Change this to control how many rows per page
    total_pages = -(-len(df_filtered) // rows_per_page)  # Ceiling division
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, value=1)

    # Get the data for the current page
    start_row = (current_page - 1) * rows_per_page
    end_row = start_row + rows_per_page
    df_paginated = df_filtered.iloc[start_row:end_row]
    # Display the paginated data in a custom format
    display_apartments(df_paginated)
    
#---------------------------------------------------------------------------------------------------------
#                            GET SIMILAR APARTMENT 
#---------------------------------------------------------------------------------------------------------



if st.button('More Recommended Apartments'):
    try:

        # Initialize FilteredData with selected filters
        FD = FilteredData(df_cleaned1, selected_state, selected_price, selected_bedrooms, selected_bathrooms)
        filtered_data = FD.filtered_data
        
        # Convert filtered data to DataFrame
        df_filtered = pd.DataFrame(filtered_data)

        # Ensure necessary columns exist
        if 'bathrooms' not in df_filtered.columns:
            st.error("The 'bathrooms' column is missing. Please check the data.")
            st.stop()
        if 'bedrooms' not in df_filtered.columns:
            st.error("The 'bedrooms' column is missing. Please check the data.")
            st.stop()

        # Calculate scores using ScoreDistribution
        bathroom_dis = ScoreDistribution(df_filtered['bathrooms'], selected_bathrooms, bathroom_rate)
        bedroom_dis = ScoreDistribution(df_filtered['bedrooms'], selected_bedrooms, bedroom_rate)
        np_score = bedroom_dis.apply_score() + bathroom_dis.apply_score()

        # Add score to the DataFrame and sort by score
        df_filtered['score'] = np_score
        df_filtered = df_filtered.sort_values(by='score', ascending=False)

        # Ensure df_filtered is not empty before proceeding
        if df_filtered.empty:
            st.warning("No recommended apartments match the selected criteria.")

        # Perform PCA and get similar apartments
        PPP = PCA_PAIRWISE(df_cleaned1)
        top5_indices = df_filtered.index[:5]
        top_similar = PPP.get_pairwise_dis(top5_index=top5_indices)

        # Display the recommended apartments
        display_apartments(df_cleaned1.loc[top_similar])
    
    except KeyError as e:
        st.error(f"A required column is missing: {e}")
    except ValueError as e:
        st.error(f"Value error encountered: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

import streamlit as st
from app import df_cleaned1, display_apartments
from utils.jagath import Simple_Search
from sklearn.metrics.pairwise import cosine_similarity

ss = Simple_Search(df_cleaned1)

@st.cache_data
def CV_matrix():
    return ss.X

@st.cache_resource
def vectorizer_cv():
    return ss.cv

st.title('Apartment Search filter')

search_query = st.text_input('Enter the Apartment Name or Address....')

if search_query:
    idxs = ss.get_top5_indices(search_query)
    try:
        idxs = [i for i in idxs if i in df_cleaned1.index]
        data = df_cleaned1.iloc[idxs,:]
        display_apartments(data)
    except:
        st.write('No relevant Apartments')
    

else:
    st.write('Please enter Name or Address of Apartment you are looking for.....')

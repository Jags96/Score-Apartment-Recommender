import re
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

################################### Clean and create_half_bedrooms ################################

def Clean(df):
    """
    Here, the DataFrame df column 'currency' is used as means for checking for 
    incorrect data entries
    """
    clean_df = df.copy()
    clean_df = clean_df[clean_df['currency']=='USD']
    
    clean_df.drop_duplicates(keep = 'first',inplace = True)
    if 'bathrooms' in clean_df.columns:
        clean_df['bathrooms'] = clean_df['bathrooms'].apply(lambda x: float(x))
    if 'bedrooms' in clean_df.columns:
        clean_df['bedrooms'] = clean_df['bedrooms'].apply(lambda x: float(x))
    if 'cats_allowed' not in clean_df.columns:
        clean_df['cats_allowed'] = clean_df['pets_allowed'].isin({'Cats,Dogs', 'Cats'})
    if 'dogs_allowed' not in clean_df.columns:
        clean_df['dogs_allowed'] = clean_df['pets_allowed'].isin({'Cats,Dogs', 'Dogs'})

    clean_df = create_half_bathrooms(clean_df)
    clean_df = clean_df.fillna({'bathrooms':0, 'bedrooms':0, 'state':'ZZ'})

    print(f'Data cleaning is success, returning clean_df')
    return clean_df

def create_half_bathrooms(df):
    if 'half_bathrooms' not in df.columns:
        df['half_bathrooms'] = 0.0
        for r in df.index:
            try:
                if df.loc[r,'bathrooms']%1 != 0:
                    df.loc[r,'half_bathrooms'] = 1
                    df.loc[r,'bathrooms'] = float(int(df.loc[r,'bathrooms']))
            except:
                continue
    return df

############################# FilteredData class ################################

class FilteredData:
    def __init__(self,main_df,selected_state,selected_price,selected_bedrooms, selected_bathrooms):
        """
        Simple class for filtering Dataframe , built just to remove unnecessary repetitive filtering of same features 

        Args:
            main_df (pd.DataFrame_): the main dataframe for filtering
            selected_state (str): selected state
            selected_price (list): selected a list of price range as [min,max]
            selected_bedrooms (int): int bedrooms
            selected_bathrooms (int): int bathrooms
        """
        self.main_df = main_df
        self.selected_state = selected_state
        self.selected_price = selected_price
        self.selected_bedrooms = selected_bedrooms
        self.selected_bathrooms = selected_bathrooms
        self.filtered_data = self._filter_function()

    def _filter_function(self):
        """_summary_

        Returns:
            pd.Dataframe: filtered Dataframe in same format as main_df
        """

        filtered_data = self.main_df[
            (self.main_df['state'] == self.selected_state) &
            #(self.main_df['bedrooms'] == self.selected_bedrooms)&
            (self.main_df['price']>=self.selected_price[0]) &
            (self.main_df['price']<= self.selected_price[1])
            #&(self.main_df['bathrooms']== self.selected_bathrooms)
        ]
        return filtered_data
####################################### Score Distribution class #######################

### Self note handling zero classes situation is pending
class ScoreDistribution:
    def __init__(self, series, primary, rate):
        """
        Initializes ScoreDistribution class with a pandas series, primary choice, and rating.
        Converts every value in the series to float and stores the unique sorted classes.

        Args:
            series (pandas.Series): Input data series
            primary (int): Index of the primary choice class
            rate (int): Rating value between 1 and 10

        Raises:
            AssertionError: If the input is not a pandas series or the rate is not between 1 and 10
        """
        assert isinstance(series, pd.core.series.Series), 'InputError: Input is not a pandas series'
        assert 1 <= rate <= 10, f'InputError: {rate} is not within the range of 1 to 10'

        self.series = series
        self.primary = primary
        self.rate = rate
        self.classes = (series.apply(float)).unique()
        self.classes.sort()
        self.num_classes = int(max(max(self.classes),self.primary)+1)
        self.final_distribution = self._get_final_rate_distribution()

    def _get_rating_distribution(self):
        """
        Calculates the rating distribution using the idea of a 10-point distribution
        around the primary choice, with a maximum limit of 2 for non-chosen classes.

        Args:
            rate (int): Rating value between 1 and 10

        Returns:
            numpy.ndarray: Rating distribution array
        """
        if self.rate == 10:
            return np.array([10])
        if self.rate == 1:
            return np.ones(10)

        distribution = [self.rate]
        flag = 1
        while sum(distribution) < 10:
            if flag == 1:
                if distribution[0] == 2 or (distribution[0] == self.rate):    ## dis[0]!=rate
                    distribution.insert(0, 1)
                elif distribution[0] == 1:
                    distribution[0] += 1
                elif distribution[0]==self.rate:
                    distribution.insert(1,1)
            elif flag == -1:
                if distribution[-1] == self.rate or (distribution[-1] >= 2):
                    distribution.append(1)
                elif distribution[-1] == 1:
                    distribution[-1] += 1
            flag *= -1

        return np.array(distribution)
    
    def _get_final_rate_distribution(self):
        """
        Calculates the final rate distribution by slicing the rating distribution
        based on the primary choice index.

        Returns:
            numpy.ndarray: Final rate distribution array
        """
        distribution = self._get_rating_distribution()
        final_distribution = np.append(np.zeros(self.num_classes), distribution)
        final_distribution = np.append(final_distribution, np.zeros(self.num_classes))
        return final_distribution[len(final_distribution) // 2 - self.primary: len(final_distribution) // 2 - self.primary + self.num_classes+1]
    
    def apply_score(self):
        """
        Applies the final rate distribution to the input data series.

        Returns:
            numpy.ndarray: Array of scores
        """
        assert isinstance(self.series, pd.core.series.Series), 'InputError: Input is not a pandas series'
        np_series_value = self.series.values
        np_series_score = np.array([self.final_distribution[int(i)] for i in np_series_value])
        #print(f'np_array of scores {np_series_score} for classes {self.classes}')
        return np_series_score
    
############################################## PCA and Pairwise matrix ###############

COLUMNS_CONSIDERED = ['bathrooms', 'bedrooms', 'price', 'square_feet', 'state', 'latitude', 'longitude', 'cats_allowed', 'dogs_allowed']
class PCA_PAIRWISE:
    def __init__(self, clean_df):
        """_Takes Apartment Dataframe to calculate PCA with two features and later can be used to get pairwise distances

        Args:
            clean_df (_type_): a clean_df is preferred and Columns considered are ['bathrooms', 'bedrooms', 'price',
              'square_feet', 'state', 'latitude', 'longitude', 'cats_allowed', 'dogs_allowed'] for PCA calculation
        """
        self.clean_df = clean_df.copy()
        self.pcadf = self._get_df_numeric_columns()
        self.indices = self.pcadf.index
        self.new_df = self._perform_pca(self.pcadf)

    def _get_df_numeric_columns(self):
        """
        Take a DataFrame and return numeric columns of the dataframe

        Returns:
          Numerics columned DataFrame
        """
        pcadf = self.clean_df
        pcadf = pcadf.fillna({'bathrooms':0., 'bedrooms':0., 'state':'NAN'})
        all_columns = pcadf.columns
        for c in all_columns: 
            if c not in COLUMNS_CONSIDERED:
                try:
                    pcadf.drop(columns = [c], inplace = True)
                except:
                    print(f' Failed to drop {c} column ')
                    continue
        pcadf.dropna(inplace = True)
        pcadf = pd.get_dummies(pcadf,columns = ['state'],drop_first = True)
        return pcadf


    def _perform_pca(self,pcadf):
        """
        Takes numeric columns Dataframe to return PCA dataframe

        Args:
            pcadf (pandas.DataFrame):  only float(or int) columned Dataframe 

        Returns:
            PCA dataframe with two features
        """
        pca = PCA()
        new_np= pca.fit_transform(pcadf)
        #sigma_variance = pca.explained_variance_ratio_
        new_np = new_np[:,:2]   #only two new features considered
        new_df = pd.DataFrame(new_np, index = self.indices, columns = ['f1','f2'])
        return new_df

    def get_pairwise_dis(self,new_df = None, top5_index = None,return_paird = False, req_top = 5):
        """
        Calculates pairwise distances and return indices of the top req_top indices 

        Args:
            new_df (_type_, optional): Dataframe on which pairwise is performed. Defaults to PCA dataframe from object.
            top5_index (_type_, optional): Pairwise distances with respect to top5_index is provided. Defaults to None.
            return_paird (bool, optional): True when Pairwise distances is required. Defaults to False.
            req_top (int, optional): _description_. Defaults to 5.

        Returns:
            Index object of req_top indices
        """
        if new_df is None:
            new_df = self.new_df
        if top5_index is None:
            top5_index = new_df.index
        pair_d = pairwise_distances(new_df, new_df.loc[top5_index], n_jobs=-1)
        ## sum(axis = 1) or along columns
        temp = pd.DataFrame(pair_d, index = self.indices, columns = top5_index )
        temp['sum'] = temp.sum(axis = 1)
        temp = temp.sort_values(by = 'sum')
        ## sort them and get top
        if return_paird:
            return pair_d
        return temp.index[:req_top]



############################################ Simple Search based on column names #########

class Simple_Search():
    def __init__(self, df, column_name = ['title','address','cityname']):
        """
        Dataframe is required when initializing Simple_Search class, columns are optional

        Args:
            df (pd.DataFrame): df on which Simple Search needs to be performed
            column_name (list, optional):  Defaults to ['title','address','cityname'].
        """
        self.df = df.copy()
        self.indices = self.df.index
        self.column_name = column_name
        self._clean_column_text()
        self._cv_matrix()
 
    def _clean_column_text(self):
        """
        Removed special characters and combines all columns mentioned when initializing the object
        """
        if isinstance(self.column_name, list):
            self.df.loc[:,self.column_name] = self.df.loc[:,self.column_name].apply(lambda s: re.sub(r'[$,.:\/|%&*]','',s) if isinstance(s, str) else s)
        else:
            self.df.loc[ :,[self.column_name]] = self.df.loc[:,[self.column_name]].apply(lambda s: re.sub(r'[$,.:/\|$%&]', '', s) if isinstance(s, str) else s )
        try:
            self.df['all_text'] = self.df.apply( lambda row: f"{row['title']} "
                                        f"{row['address'] if pd.notna(row['address']) else ''} "
                                        f"{row['cityname'] if pd.notna(row['cityname']) else ''}",
                                        axis=1)
            self.column_name = 'all_text'
        except:
            self.column_name = 'title'
 
    def _cv_matrix(self):
        """
        fits a Count Vectorizer and stores within the class
        """
        corpus = []
        if self.column_name in self.df.columns:
            for i in self.df.index:
                corpus.append(self.df[self.column_name][i])
        self.cv = CountVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2))
        self.X =  self.cv.fit_transform(corpus)
 

    def get_top5_indices(self,text, top = 5, threshold = 0.1):
        """
        Return Indices of rows in df which are similar to text query. 

        Args:
            text (str): text query
            top (int, optional): Required number of indices. Defaults to 5.
            threshold (float, optional): cosine similairty threshold to consider returning indices. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        input_vector = self.cv.transform([text])
        scores = cosine_similarity(input_vector, self.X)
        if (scores>=threshold).sum():
            idx = scores.argsort()[0][-1:-(top+1):-1]
            return idx  
        return None
    

"""
#-------------------------------------------------------------------------
#               GET SCORE BASED TOP APARTMENTS
#-------------------------------------------------------------------------

    st.write("Score Based Top Apartments")
    # Assuming `df_filtered` is already available and contains all the columns provided
    # Ensure `df_filtered` is a pandas DataFrame
    #used current session df Jagath may change back to filtered_data
    df_filtered = pd.DataFrame(st.session_state.data[st.session_state.counter])
    ### Adding the scoring system/methods
    bathroom_dis = ScoreDistribution(df_filtered['bathrooms'], selected_bathrooms, bathroom_rate)
    bedroom_dis = ScoreDistribution(df_filtered['bedrooms'], selected_bedrooms, bedroom_rate)
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
       ##need the top5 indices for PCA calculation
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
"""

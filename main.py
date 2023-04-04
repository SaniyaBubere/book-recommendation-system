import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# loading in the model to predict on the data
pickle_in = open('model_knn.pkl', 'rb')
model_knn = pickle.load(pickle_in)


# loading in the model to predict on the data
pickle_in = open('ratings_matrix_new.pkl', 'rb')
ratings_matrix_new = pickle.load(pickle_in)


# here we define some of the front end elements of the web page like
# the font and background color, the padding and the text to be displayed
html_temp = """
<div style ="background-color:gray;padding:13px">
<h1 style ="color:black;text-align:center;"> Recommendation System </h1>
</div>
"""

# this line allows us to display the front end aspects we have
# defined in the above code
st.markdown(html_temp, unsafe_allow_html=True)

bookName = st.text_input("Book Name")
number = st.number_input('Number of Recommendation',min_value=1,max_value=10)

def prediction(book_title,number):
    result=[]

    # Finding the index of the book in ratings_matrix_new
    if book_title in ratings_matrix_new.index:# checks if the input book title is present in the index of ratings_matrix_new, which is a Pandas DataFrame containing the user ratings.
        query_index = ratings_matrix_new.index.get_loc(book_title)#gets the index of the input book title in the ratings_matrix_new.
        # result.append('Book title:', book_title)#prints the book title that was entered by the user.

        # Generating recommendations
        distances, indices = model_knn.kneighbors(ratings_matrix_new.iloc[query_index,:].values.reshape(1, -1), n_neighbors= int(number+1))# Find the indices and distances of the k-nearest neighbors for the book title
        for i in range(0, len(distances.flatten())):# Iterate over each neighbor
            if i == 0:# Skip the first neighbor as it will always be the book itself
                continue
            else:
                result.append('{}. {}'.format(i, ratings_matrix_new.index[indices.flatten()[i]]))# Print the rank, book title and distance of each neighbor
    else:
        result.append("Book title not found in dataset.")
    return result
#
if st.button("Recommend"):
    result = prediction(bookName,number)
    for i in result:
        st.success(i)


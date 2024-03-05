import pickle
import streamlit as st
import numpy as np

st.header('Book Recommendation System')
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
Book_List = pickle.load(open('Book_List.pkl', 'rb'))


def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    distances = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)

    book_data = []
    for i in distances[1:8]:
        item = []
        # print(pt.index[i[0]])
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        book_data.append(item)

    return book_data


Selected_Book = st.selectbox(
    "What would you like to",
    Book_List
)

if st.button('Show Recommended Books'):
    rec = recommend(Selected_Book)

    for recommended_book in rec:
        st.write(f"**Book Title:** {recommended_book[0]}")
        st.write(f"**Book Author:** {recommended_book[1]}")
        st.image(recommended_book[2], caption='Book Cover', width=100)
        st.write('---')  # Separator between books

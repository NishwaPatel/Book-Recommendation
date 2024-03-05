import numpy as np
import pandas as pd

books = pd.read_csv("Books.csv")
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")

print(books)
print(users)
print(ratings)

print(books.shape)
print(users.shape)
print(ratings.shape)

print(books.isnull().sum())
print(users.isnull().sum())
print(ratings.isnull().sum())

print(books.duplicated().sum())
print(users.duplicated().sum())
print(ratings.duplicated().sum())
print(ratings)

ratings_with_name = ratings.merge(books,on="ISBN")

print(ratings_with_name)
print(ratings_with_name[["Book-Title","Book-Rating"]])
print(ratings_with_name)


x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
good_users = x[x].index

filter_ratings = ratings_with_name[ratings_with_name['User-ID'].isin(good_users)]
print(filter_ratings)

y = filter_ratings.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y]
famous_books = famous_books.index
print(famous_books)

Book_List = []
for a in famous_books:
    Book_List.append(a)

print(Book_List)


final_ratings = filter_ratings[filter_ratings['Book-Title'].isin(famous_books)]
print(final_ratings)
print(final_ratings.drop_duplicates())
pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
print(pt)
pt.fillna(0,inplace=True)
print(pt)

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(pt)
print(similarity)
print(similarity[0])
print(similarity.shape)

print(pt.index[5])
print(pt.index[0])
print(np.where(pt.index == 'A Case of Need')[0][0])
print(np.where(pt.index=='Zoya')[0][0])

def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    distances = sorted(list(enumerate(similarity[index])),key=lambda x:x[1],reverse=True)
    
    book_data = []
    for i in distances[1:8]:
        item = []
        #print(pt.index[i[0]])
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        book_data.append(item)
        
    return book_data


show_re = recommend('1984')

print("-----------------")
print(show_re)

import pickle

with open('pt.pkl', 'wb') as f:
    pickle.dump(pt, f)
with open('pt.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('books.pkl', 'wb') as f:
    pickle.dump(books, f)
with open('books.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('similarity.pkl','wb') as f:
    pickle.dump(similarity, f)
with open('similarity.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('Book_List.pkl','wb') as f:
    pickle.dump(Book_List, f)
with open('Book_List.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

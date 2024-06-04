import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def fomat_tittle(title):
    return title.replace("|", " ").replace("-", " ")


data = pd.read_csv("movies.csv", encoding='latin1', sep='\t', usecols=['title', 'genres'])
data["genres"] = data['genres'].apply(lambda title: title.replace("|", " ").replace("-", ""))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['genres'])
tfidf_matrix_dense = pd.DataFrame(vectorizer.fit_transform(data['genres']).todense(), index=data["title"],
                            columns=vectorizer.get_feature_names_out())

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_dense = pd.DataFrame(cosine_sim, index=data["title"], columns=data["title"])


input_movie = "King and I, The (1999)"
top =20

resul_movie = cosine_sim_dense.loc[input_movie, :]
resul_movie = resul_movie.sort_values(ascending=False)[:top].reset_index()

print(resul_movie[['title']])
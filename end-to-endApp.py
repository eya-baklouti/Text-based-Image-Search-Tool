import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer 
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import numpy as np
import itertools

st.title("Illustrate My Text")

placeholder = st.empty()

text = placeholder.text_area("Write your story or artical", height=300)

top_n = st.sidebar.number_input("Select number of keywords to extract",1, 10, 1, 1)
n_image = st.sidebar.number_input("Select number of image to display ",1, 10, 1, 1)


check1 = st.sidebar.checkbox('Keywords')
check2 = st.sidebar.checkbox('Diversified keywords using maximum sum similarity')
check3 = st.sidebar.checkbox('Diversified keywords using maximal marginal relevance')

##############################################################
                # USEFUL FUNCTIONS #
##############################################################
def filter_numbers(candidates):
    candidates2=[]
    for i in candidates:
        
        if i.isalpha():
            candidates2.append(i)
    return candidates2


def max_sum_similarity(doc_embedding: np.ndarray,candidate_embeddings: np.ndarray,words,top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_words = cosine_similarity(candidate_embeddings, candidate_embeddings)

    # Get 2*top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    candidates = distances_words[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = 10000
    candidate = None
    k=0
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [(words_vals[idx], round(float(distances[0][idx]), 4)) for idx in candidate]


def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def search(query, k=n_image):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]
    
    #display(query)
    for hit in hits:

        image = Image.open('C:\\Users\\Asus\\Documents\\Multi media retrievel\\photos\\'+str(img_names[hit['corpus_id']]))
        st.image(image)

#############################################################
          # KEY WORD EXTRACTION #
###############################################################


#Text processing 
# Extract candidate keywords
count = CountVectorizer(ngram_range=(1, 1), stop_words="english").fit([text])
candidates = count.get_feature_names()


#we remove numbers because generally we don't need them in the keywords thqt relqted to images
candidates=filter_numbers(candidates)

#lematization in candidates
lemmatizer = WordNetLemmatizer()
can=set()
for i in candidates:
    can.add(lemmatizer.lemmatize(i)) 

candidates=list(can)

#keywords embedding
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([text])
candidate_embeddings = model.encode(candidates)



##############################################################
            # SIMILARITY CALCULATION #
#################################################################

# cosine similarity calculation 
top_nk = 10 #the number of keywords that we want to preselect
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_nk:]]


################## ADDING DIVERSIFICATION ######################

keywords_embeddings = model.encode(keywords)

# adding the max sum similarity diversification
nr_candidates=len(keywords)
max_sum_sim_list=max_sum_similarity(doc_embedding, keywords_embeddings, keywords, top_n, nr_candidates)


# adding the mmr similarity diversification
mmr_list=mmr(doc_embedding, keywords_embeddings, keywords, top_n, 0.2)

################## DISPLAYING KEY WORDS ########################
max_sum_sim_listd=[]
for i in range(len(max_sum_sim_list)):
    max_sum_sim_listd.append(max_sum_sim_list[i][0])
            
            
keywords2=keywords[:5]
if (keywords2 != []) and (mmr_list != []) and (max_sum_sim_list != []):
    keywordsd = pd.DataFrame(list(zip(keywords2,mmr_list,max_sum_sim_listd)),columns=["keyword", "keywords with mmr diversification","keywords with max sum sim diversification"])
    st.table(keywordsd)



#####################################################################
                       # IMAGE SEARCH #
###################################################################


torch.set_num_threads(4)
img_folder = 'photos/'

#image text embedding model
model = SentenceTransformer('clip-ViT-B-32')



# used precomputed embeddings here just to make the results come faster
emb_filename = 'unsplash-25k-photos-embeddings.pkl'
with open(emb_filename, 'rb') as fIn:
    img_names, img_emb = pickle.load(fIn)  
    
#display images 
if check1:
    st.markdown("Images found just using the keyword")
    s=""
    for k in keywords2:
        s=s+" "+k
    search(s)


if check2:
    st.markdown("Images found using the keywords with max sum sim diversification")
    s=""
    for k in max_sum_sim_listd:
        s=s+" "+k
    search(s)

if check3:
    st.markdown("Images found using the keywords with mmr diversification")
    s=""
    for k in mmr_list:
        s=s+" "+k
    search(s)



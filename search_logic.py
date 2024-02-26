import pickle
import string
import spacy
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")


def custom_tokenizer(doc):
    return [t.lemma_ for t in nlp(doc) if (not t.is_punct) and (not t.is_stop)]


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_results(query):
    with open("data/faculty_research_combined.pkl", "rb") as fp:
        faculty_research = pickle.load(fp)

    # Set up corpus with vectors for faculty
    # words = set()
    corpus = []
    for key in faculty_research:
        text = faculty_research[key]
        text = text.replace(".", ". ")
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        corpus.append(text)
        # print(key, text)
        # for punc in string.punctuation:
        #   text = text.replace(punc, ' ')
        # split = text.split()
        # print(split)
        # words.update(split)
    print()
    # print(words)

    # print(corpus)

    # doc = nlp(combined)
    # print([t.text for t in doc])

    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, binary=True)
    bow = vectorizer.fit_transform(corpus)
    dictionary = vectorizer.vocabulary_
    # print(dictionary)
    # print()
    # print(bow.toarray())

    # query = "machine learning"
    query_vectorizer = CountVectorizer(tokenizer=custom_tokenizer, binary=True)
    query_list = []
    query_list.append(query)
    query_bow = query_vectorizer.fit_transform(query_list)
    query_terms = query_vectorizer.vocabulary_
    # print(query_terms)
    query_vector = np.array([0] * len(dictionary))
    for key in query_terms:
        if key in dictionary:
            query_vector[dictionary[key]] = 1
    # print(query_vector)

    # print(query)
    # print(query_vector)
    # print()

    cosine_similarities = {}

    for i in range(0, len(corpus)):
        sim = cosine_sim(query_vector, bow[i].toarray().squeeze())
        if sim > 0:
            cosine_similarities[i + 1] = sim
            # print(corpus[i])
            # print(bow[i].toarray().squeeze())
            # print(f'Similarity score: {sim:.3f}')
            # print()

    # Sort cosine sim dict
    sorted_cosine_sim = sorted(
        cosine_similarities.items(), key=lambda x: x[1], reverse=True
    )
    cosine_sim_sorted = dict(sorted_cosine_sim)
    # print(cosine_sim_sorted)

    # Retrieve all info about the top 20 ranked faculty in cosine_sim_sorted

    # name, link
    with open("data/faculty_info.pkl", "rb") as fp:
        faculty_info = pickle.load(fp)
        # print('Faculty info dictionary')
        # print(faculty_info)

    # appt, research, bio, current research
    with open("data/faculty_research.pkl", "rb") as fp:
        faculty_research = pickle.load(fp)
        # print('Faculty research dictionary')
        # print(faculty_research)

    i = 0
    results = []
    for key in cosine_sim_sorted:
        if i >= 20:
            break
        id = key
        sim = cosine_sim_sorted[key]
        faculty_info_list = faculty_info[id]
        faculty_research_list = faculty_research[id]
        results.append(
            {
                "Name": faculty_info_list[0],
                "Link": faculty_info_list[1],
                "Department": faculty_research_list[0],
                "Research": faculty_research_list[1],
                "Relevance": f"{sim:.2f}",
            }
        )
        # print("Name:", faculty_info_list[0])
        # print("Link:", faculty_info_list[1])
        # print("Department:", faculty_research_list[0])
        # print("Research:", faculty_research_list[1])
        # print(f"Relevance: {sim:.2f}")
        # print()
        i += 1
    # print(results)
    return results

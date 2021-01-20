import nltk
import stopwords as stopwords

from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
nltk.download('stopwords')

def read_article(file):
    fileread = open(file, "r")
    fileData = fileread.readlines()
    article = fileData[0].split(".")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split())
    sentences.pop()
    return sentences

def sentence_similarity(sentence1, sentence2, stopwords = None):
    if stopwords is None:
        stopwords = []
    sentence1 = [w.lower() for w in sentence1]
    sentence2 = [w.lower() for w in sentence2]
    allwords = list(set(sentence1+sentence2))
    vector1 = [0]*len(allwords)
    vector2 = [0]*len(allwords)
    for w in sentence1:
        if w in stopwords:
            continue
        vector1[allwords.index(w)] += 1
    for w in sentence2:
        if w in stopwords:
            continue
        vector2[allwords.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def generating_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])
    return similarity_matrix

def generate_summary(file, top_n= 5):
    stop_words = stopwords.get_stopwords('english')
    summarize_text = []
    sentences = read_article(file)
    sentence_similarity_matrix = generating_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse = True)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("Summary: \n",". ".join(summarize_text))
generate_summary("Astronomy.txt", 2)
#the text is from the Atlantic Article "Astronomers Are Keeping a Close Watch on the Next Star Over" by Marina Koren









# Press the green button in the gutter to run the script.
#if __name__ == '__main__':



import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, \
    manhattan_distances, cosine_distances
from rank_bm25 import BM25Okapi

    
def bm25_similarity(queries, corpus, metric_params):
    
    tokenized_corpus = [doc.split() for doc in corpus]
    tokenized_queries = [list(set(doc.split())) for doc in queries]
    bm25 = BM25Okapi(tokenized_corpus, k1=metric_params['k1'], \
        b=metric_params['b'])
    
    bm25_score = np.zeros(shape=(len(tokenized_queries), \
        len(tokenized_corpus)))
    
    for idx, query in enumerate(tokenized_queries):
        bm25_score[idx] = bm25.get_scores(query)
    
    # Normalize bm25 simmilarity values
    max_score = np.max(bm25_score)
    bm25_score_norm = (bm25_score - 0) / (max_score - 0)

    return bm25_score_norm

def compute_distances(X, Y, metric, metric_params):
    
    if metric == 'manhattan':
        distances = manhattan_distances(X, Y)
    elif metric == 'euclidean':
        distances = euclidean_distances(X, Y)
    elif metric == 'cosine':
        distances = cosine_distances(X, Y)
    elif metric == 'bm25':
        distances = bm25_similarity(X, Y, metric_params)
    
    return distances

def convert_distances_to_similarities(distances, metric):
    
    # Normalize distances values
    if metric == 'cosine':
        # cosine distances has 1.0 for maximum and 0 for minimum values
        distances_norm = distances
    else:
        max_distances = np.max(distances)
        min_distances = np.min(distances)
        distances_norm = (distances - min_distances) / (max_distances - min_distances) 

    # Convert distances values into similarities by subtracting 1 
    # with the distances values
    simmilarities = 1.0 - distances_norm
    return simmilarities

def compute_similarities(X, Y, metric, metric_params):
    
    distances = compute_distances(X, Y, metric, metric_params)
    
    similarities = convert_distances_to_similarities(distances, metric) \
        if metric!='bm25' else distances
    
    return similarities


def compute_class_score(class_label, k_neighbors):
    
    
    return class_score

def nearest_neighbors(similarities, class_label, num_k):
    similarity_label_pairs = [*zip(similarities, class_label)]
    # Sorting similarity_label_pairs into descending order
    similarity_label_pairs = sorted(similarity_label_pairs, \
        key= lambda x: x[0], reverse=True)

    return similarity_label_pairs[:num_k]

def predict_class(similarities, class_label, num_k):
    
    k_neighbors = nearest_neighbors(similarities, class_label, num_k)

    # Generate pair of class label and its score
    k_neighbors = np.array(k_neighbors)
    label, score = np.unique(k_neighbors[:, 1], return_counts=True)
    label_score_pairs = [*zip(label, score)]
    
    # Return class label with the maximum class score
    return max(label_score_pairs, key=lambda x: x[1])[0]

def compute_num_n(class_freq, num_k):
    
    max_freq = max(class_freq.values())
    num_n = {}
    for label in class_freq:
        num_n[label] = int(np.ceil((num_k*class_freq[label]) / max_freq))
    
    return num_n

def compute_class_score_iknn(class_label, n_neighbors):
    
    n_neighbors = np.array(n_neighbors)
    
    score, total_sim = 0, 0
    for similarity, label in n_neighbors:
        if class_label == label:
            score += similarity
        total_sim += similarity
        
    return score / total_sim

def predict_class_iknn(similarities, class_label, num_n):
    
    class_score = []
    for class_ in num_n:
        n_neighbors = nearest_neighbors(similarities, class_label, num_n[class_])
        class_score.append((class_, compute_class_score_iknn(class_, \
            n_neighbors)))
    
    # Return class with the maximum class score
    return max(class_score, key= lambda x: x[1])[0]

def compute_nwknn_weight(class_freq, exponent):
    
    min_freq = min(class_freq.values())
    class_weight = {}
    for label in class_freq:
        class_weight[label] = 1 / ((class_freq[label]/min_freq)**(1/exponent))

    return class_weight

def predict_class_nwknn(similarities, class_label, num_k, weight_class):
    
    k_neighbors = nearest_neighbors(similarities, class_label, num_k)

    # Generate pair of class label and its score
    k_neighbors = np.array(k_neighbors)
    label, score = np.unique(k_neighbors[:, 1], return_counts=True)
    label_score_pairs = [(key, weight_class[key]*val) \
        for key, val in zip(label, score)]
    
    # Return class label with the maximum class score
    return max(label_score_pairs, key=lambda x: x[1])[0]
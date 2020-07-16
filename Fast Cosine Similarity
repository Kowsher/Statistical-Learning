from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.preprocessing import normalize
A_sparse = sparse.csr_matrix(train)
#B_sparse = sparse.csr_matrix(B)



kk = np.dot(normalize(A_sparse, axis=1), normalize(A_sparse, axis=1).T)

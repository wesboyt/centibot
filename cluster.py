import faiss
import json
from sentence_transformers import SentenceTransformer

comments = json.loads(open("comments.json", "r").read())
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
x = model.encode(comments)
d = x.shape[1]
index = faiss.read_index("submissions.index")
ncentroids = 1024
niter = 100
verbose = True
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, nredo=200)
kmeans.train(x)
D, i = kmeans.index.search(x, 1)
f = open("clusters.json", "w")
f.write(json.dumps(i.tolist()))
f.close()

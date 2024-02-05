from .recoverer import ClusterRepRecoverer
from .utils import Clustering

cache_dir = ".cache/"
lm_num_possibilities = 10
clustering = Clustering.from_pickle("../../clusterers/agglomerative_cluster_gamma0.3.pkl",
                                    max_num_possibilities=None)
recoverer = ClusterRepRecoverer(cache_dir, clustering)

# Interactive loop for testing recoverer
while True:
    text = input("Enter text: ")
    if text == 'q':
        break
    print("\nRecovered text: {}\n".format(recoverer.recover(text)))

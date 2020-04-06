import h5py

embeddings = []
with h5py.File("embeddings.h5", 'r') as hf:
    grp = hf["embeddings"]
    embeddings = [e.value for e in grp.values()]

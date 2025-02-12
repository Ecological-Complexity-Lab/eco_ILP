from node2vec import Node2Vec
import pandas as pd
import gensim

def train_node2vec(B):
    g_emb = Node2Vec(B, dimensions=64, quiet=True)
    model = g_emb.fit(
        vector_size=64,  # Changed to match the dimensions argument
        window=1,
        min_count=1,
        batch_words=4
    )
    model.save("models/node2vec_model.model")  # Save the model
    return model

def generate_embeddings(B, model):
    emb_df = pd.DataFrame(
        [model.wv.get_vector(str(n)) for n in B.nodes() if str(n) in model.wv],
        index=[n for n in B.nodes() if str(n) in model.wv]
    )
    return emb_df

# Usage
model = train_node2vec(B)
emb_df = generate_embeddings(B, model)

def load_and_embed_new_network(new_B):
    model = gensim.models.Word2Vec.load("models/node2vec_model.model")
    new_emb_df = generate_embeddings(new_B, model)
    return new_emb_df

# Usage
new_emb_df = load_and_embed_new_network(new_B)

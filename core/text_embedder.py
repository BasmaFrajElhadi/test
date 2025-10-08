from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class TextEmbedder():

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device = "cpu", dim = 384):
        self.model_name = model_name
        self.device = device
        self.dim = dim

    def embedding(self):
        return SentenceTransformerEmbeddingFunction()
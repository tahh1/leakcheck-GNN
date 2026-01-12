import threading
import sent2vec
from tokenizers import Tokenizer

class ModelLoader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load the tokenizer and model
        tokenizer_path = "./src/tokenizer_normalized.json"
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        self.sent2vec_model = sent2vec.Sent2vecModel()
        self.sent2vec_model.load_model("./src/my_model_normalized_meths_32.bin")


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
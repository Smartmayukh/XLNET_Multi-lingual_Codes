import transformers

MAX_LEN=512
TRAIN_BATCH_SIZE=8
VALID_BATCH_SIZE=4
EPOCHS=15
ACCUMULATION=2
BERT_PATH=["bert-base-multilingual-cased","bert-base-multilingual-uncased","monsoon-nlp/hindi-bert","monsoon-nlp/hindi-tpu-electra"]
MODEL_PATH=[]
TRAINING_FILE="/content/Sentmodel-Hindi/inputs/review.csv"
TOKENIZER=[transformers.AutoTokenizer.from_pretrained (BERT_PATH[i]) for i<len(BERT_PATH)]
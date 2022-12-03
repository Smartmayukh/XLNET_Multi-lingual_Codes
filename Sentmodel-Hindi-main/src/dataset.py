import config
import transformer
import os

MODEL=os.environ.get("MODEL")

tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL)
class HindiReviewDataset(Dataset):
  def __init__(self, text, polarity, tokenizer,max_len):
    self.text = text
    self.polarity = polarity
    self.tokenizer = config.tokenizer
    self.max_len=config.max_len
  def __len__(self):
    return len(self.text)
  def __getitem__(self, item):
    review = str(self.text[item])
    target = self.polarity[item]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


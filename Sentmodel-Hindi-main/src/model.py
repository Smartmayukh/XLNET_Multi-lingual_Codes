import transformers
import torch.nn as nn
import os 

MODEL=os.environ.get("MODEL")

model_name=dispatcher.BERT_PATH[MODEL]
class SentimentClassifier(nn.Module):
  def __init__(self,model_name,n_classes):
    super(SentimentClassifier,self).__init__()
    self.bert=BertModel.from_pretrained(model_name)
    self.drop=nn.Dropout(p=0.3)
    self.linear=nn.Linear(self.bert.config.hidden_size,n_classes)
    self.softmax=nn.Softmax(dim=1)
  
  def forward(self,input_ids,attention_mask):
    outputs=self.bert(input_ids=input_ids,attention_mask=attention_mask)
    pooled_output=outputs['pooler_output']
    #Try out different things like average pooling max pooling here
    output=self.drop(pooled_output)
    output=self.linear(output)
    return self.softmax(output)
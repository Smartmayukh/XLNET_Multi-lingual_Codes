from . import dispatcher
import model

import torch.nn as nn

optimizer=AdamW(model.parameters(), lr=2e-5, correct_bias=False, ) #Used in original Bert paper
total_steps=len(train_data_loader)*EPOCHS
scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
loss_fn=nn.CrossEntropyLoss().to(device)

def train_epoch(
    model, data_loader,loss_fn, optimizer, device, scheduler, n_examples
):
 model=model.train()
 losses=[]
 correct_predictions=0

 for bi,d in enumerate(data_loader):
  input_ids=d['input_ids'].to(device)
  attention_mask=d['attention_mask'].to(device)
  targets=d['targets'].to(device)
  outputs=model(input_ids=input_ids,attention_mask=attention_mask)
  _,preds=torch.max(outputs,dim=1)
  loss=loss_fn(outputs,targets)
  correct_predictions+=torch.sum(preds==targets)
  losses.append(loss.item())
  loss.backward()
  nn.utils.clip_grad_norm(model.parameters(),max_norm=0.1)
  #If memory error, consider using gradient accumulation here
  optimizer.step()
  scheduler.step()
  optimizer.zero_grad()
 return correct_predictions.double()/n_examples, np.mean(losses)




def eval_model(model,data_loader,loss_fn,device,n_examples):
  model=model.eval()
  losses=[]
  correct_predictions=0
  with torch.no_grad():
    for d in data_loader:
      input_ids=d['input_ids'].to(device)
      attention_mask=d['attention_mask'].to(device)
      targets=d['targets'].to(device)
      outputs=model(input_ids=input_ids,attention_mask=attention_mask)
      _,preds=torch.max(outputs,dim=1)
      loss=loss_fn(outputs,targets)
      correct_predictions+=torch.sum(preds==targets)
      losses.append(loss.item())
  
  
  return correct_predictions.double()/n_examples, np.mean(losses)

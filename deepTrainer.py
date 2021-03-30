import time
import torch
import numpy as np
import gc
def torch_trainer(epochs,model,optimizer,loss_fn,trainLoader, validLoader,device):
    model.to(device)
    best_accuracy = 0
    print('Start training.....\n')
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val ACC':^9} | {'Elapsed':^9}")
    print("-"*60)

    for epoch_i in range(epochs):


        total_loss = 0
        t0_epoch = time.time()
        model.train()

        for batch in trainLoader:
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass
            b_input_ids = batch[0].to(device).long()
            b_labels = batch[1].type(torch.LongTensor).to(device)

            model.zero_grad()
            logits = model(b_input_ids)
            b_labels = b_labels.long()
            loss = loss_fn(logits,b_labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(trainLoader)

        if validLoader is not None:
            val_loss,val_accuracy, preds = evaluate(model,validLoader,loss_fn, device)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        else:
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {0.0: ^ 10.6f} | {0.0: ^ 9.2f} | {time_elapsed: ^ 9.2f}")
    print('\n')

def evaluate(model, validationLoader, loss_fn, device):
    model.eval()
    val_accuracy = []
    val_loss = []
    Predictions = []

    for batch in validationLoader:
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        b_input_ids, b_labels = batch[0].to(device), batch[1].type(torch.LongTensor).to(device)
        with torch.no_grad():
            logits = model(b_input_ids)
        loss = loss_fn(logits,b_labels)
        val_loss.append(loss)

        preds = torch.argmax(logits,dim=1).flatten()
        Predictions.extend(list(preds))
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy, Predictions
def predict(model, testLoader, device):
    Predictions = []
    model.eval()
    for batch in testLoader:
        b_input_ids, b_labels = batch[0].to(device), batch[1].type(torch.LongTensor).to(device)
        with torch.no_grad():
            logits = model(b_input_ids)
            preds = torch.argmax(logits, dim=1)
        Predictions.extend(preds)
    return Predictions



from preProcessor import *
from transformerTrainer import *
from transformerModels import CNN, LSTM
class transformerClassifier():
    def __init__(self,config):
        self.config = config

    def createModel(self, n_classes):
        if self.config['deepModel_type'] == 'cnn':
            model = CNN(self.config,  n_classes)
        elif self.config['deepModel_type'] == 'lstm':
            model = LSTM(self.config,  n_classes)
        else:
            model = None
        return model
    def trainModel(self,model,optimizer,loss_fn,trainLoader, validLoader,device):
        bert_trainer(model, trainLoader, loss_fn, optimizer, device, val_dataloader=validLoader, epochs=self.config['epochs'], evaluation=True)
    def evaluateModel(self):
        pass
    
    def saveModel(self, model):
        if self.config['save_model']:
            torch.save(model,self.config['path_to_save_model'])
        else:
            pass

    def loadModel(self):
        PATH = self.config['path_to_model']
        try:
            model = torch.load(PATH)
            model.eval()
        
            return model
        except:
            print('The path to load the model is invalid!!!!!')
    def writePrediction(self, Predictions):
        if self.config['write_predictions'] :
            with open(self.config['path_to_write_predictions'],'w') as o:
                for e in Predictions:
                    o.write(str(e.item())+'\n')
        else:
            pass
    
    def predict(self,model, test_dataloader,device):
        model.to(device)
        model.eval()
        predictions = []

        for batch in test_dataloader:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                logits = model(b_input_ids,b_attn_mask)
                preds = torch.argmax(logits,dim = 1).flatten()
            predictions.extend(preds)
        
        self.writePrediction(predictions)
        return predictions
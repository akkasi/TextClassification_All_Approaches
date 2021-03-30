from preProcessor import *
from deepModels import CNN, LSTM
from deepTrainer import *
class deepClassifier:
    def __init__(self, config):
        self.config = config

    def createModel(self,  embedding, n_classes):
        if self.config['deepModel_type'] == 'cnn':
            model = CNN(self.config, embedding, n_classes)
        elif self.config['deepModel_type'] == 'lstm':

            model = LSTM(self.config, embedding, n_classes)
        else:
            model = None
        return model

    def trainModel(self,model,optimizer,loss_fn,trainLoader, validLoader,device):
        torch_trainer(self.config['epochs'], model, optimizer, loss_fn, trainLoader, validLoader, device)


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


    
    def predict(self, model, testLoader, device):
        model.to(device)   
        model.eval()
        Predictions = []
        for batch in testLoader:
            gc.collect()
            torch.cuda.empty_cache()
            b_input_ids, b_labels = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                logits = model(b_input_ids)

            preds = torch.argmax(logits,dim=1).flatten()
            Predictions.extend(preds)
        self.writePrediction(Predictions)
        return Predictions


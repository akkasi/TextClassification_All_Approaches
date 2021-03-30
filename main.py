import yaml
from deepClassifiers import deepClassifier
from transformerClassifiers import *
# from traditionalClassifiers import *
import torch.nn as nn
config_file = open("configure.yaml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def main():
    if config['model_type'] == 'deep':
        cls = deepClassifier(config)
        data = deepPreProcessing(config)
        model = cls.createModel(data.embeddingWeights, data.n_classes).to(device)

    elif config['model_type'] == 'transformer':
        cls = transformerClassifier(config)
        data = transPreProcessing(config)
        model = cls.createModel(data.n_classes).to(device)
 
        pass
    elif config['model_type'] == 'traditional':
        cls = traditionalClassifier(config)
        data = tradPreProcessing(config)
        pass

    else:
        print('The model_type is not valid inside confif.yaml file!!!')
        exit()

    if config['choice'] == 'train':
        trainLoader = data.dataLoader(data.train)
        testLoader = data.dataLoader(data.test,shuffle=False)
        validLoader = data.dataLoader(data.valid)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(),lr = float(config['lr']))
        model.to(device)
        
        cls.trainModel(model,optimizer,loss_fn,trainLoader, validLoader,device)

        Predictions = cls.predict(model,testLoader,device)
        cls.saveModel(model)
    elif config['choice'] == 'test':
        testLoader = data.dataLoader(data.test)
        model = cls.loadModel()
        cls.predict(model,testLoader,device)
    else:
        print('There is no valid choice in configure.yaml file!')
        
main()
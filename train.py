import torch
import argparse
from model import ZubyNetV3, CNNModel1
from data_loader import CustomImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
import time 
import datetime
import os
import numpy as np
import random
import wandb

np.random.seed(10)
random.seed(10)
torch.manual_seed(10)
os.environ["PYTHONHASHSEED"] = str(10)
print('Using seed 10')

class Train():
    def __init__(self, config):
        super(Train, self).__init__()



        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

        #self.model = ZubyNetV3(3, 3, 1).to(self.device)

        self.model = CNNModel1(512, 256, 0.1).to(self.device)

        self.learning_rate = config.learning_rate

        self.bs = config.bs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_set = CustomImageDataset('/Users/atabeyunlu/ZubyNet-PyTorch/data/final_dataset/train_data.csv', 
                                               '/Users/atabeyunlu/ZubyNet-PyTorch/data/final_dataset/images', transform=None, target_transform=None)

        self.val_set = CustomImageDataset('/Users/atabeyunlu/ZubyNet-PyTorch/data/final_dataset/val_data.csv', 
                                             '/Users/atabeyunlu/ZubyNet-PyTorch/data/final_dataset/images', transform=None, target_transform=None)

        self.test_set = CustomImageDataset('/Users/atabeyunlu/ZubyNet-PyTorch/data/final_dataset/test_data.csv', 
                                              '/Users/atabeyunlu/ZubyNet-PyTorch/data/final_dataset/images', transform=None, target_transform=None)
        
        self.train_loader = DataLoader(self.train_set, batch_size=self.bs, shuffle=True)

        self.val_loader = DataLoader(self.val_set, batch_size=self.bs, shuffle=True)

        self.test_loader = DataLoader(self.test_set, batch_size=self.bs, shuffle=True)

        self.epochs = config.epochs

        self.use_wandb = True
        
        self.online = True

    def print_network(self, model, name, save_dir):
            """Print out the network information."""
            num_params = 0
            for p in model.parameters():
                num_params += p.numel()

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            network_path = os.path.join(save_dir, "{}_modules.txt".format(name))
            with open(network_path, "w+") as file:
                for module in model.modules():
                    file.write(f"{module.__class__.__name__}:\n")
                    print(module.__class__.__name__)
                    for n, param in module.named_parameters():
                        if param is not None:
                            file.write(f"  - {n}: {param.size()}\n")
                            print(f"  - {n}: {param.size()}")
                    break
                file.write(f"Total number of parameters: {num_params}\n")
                print(f"Total number of parameters: {num_params}\n\n")

    def training(self):

        if self.use_wandb:
            mode = 'online' if self.online else 'offline'
        else:
            mode = 'disabled'
        kwargs = {'name': "CNNModel1", 'project': 'ZubyNetV3', 'config': config,
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode, 'save_code': True}
        wandb.init(**kwargs)


        self.start_time = time.time()
        self.model.train()
        self.print_network(self.model, "ZubyNetV3", "/Users/atabeyunlu/ZubyNet-PyTorch/saved_model")
        for epoch in range(self.epochs):
            wandb.log({"epoch": epoch+1})
            for i, (images, labels) in enumerate(self.train_loader):
                wandb.log({"iter": i+1})

                images = images.float().to(self.device)
       
                labels_onehot = torch.nn.functional.one_hot(labels, num_classes=3).float().to(self.device)
  
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                
                loss = self.criterion(outputs, labels_onehot)
 
                loss.backward()
                self.optimizer.step()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                if (i + 1) % 10 == 0:
                    et = time.time() - self.start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    acc = accuracy_score(labels, preds)
                    mcc = matthews_corrcoef(labels, preds)
                    f1 = f1_score(labels, preds, average='macro')
                    wandb.log({"loss": loss.item(), "accuracy": acc, "mcc": mcc, "f1_score": f1})
                    print('Elapsed: [{}] , Epoch [{}/{}], Step [{}/{}], Accuracy: {:.4f}, MCC: {:.4f}, F1: {:.4f}, Loss: {:.4f}'.format(et, epoch + 1, self.epochs, i + 1,
                                                                              len(self.train_loader),
                                                                                acc,
                                                                                mcc,
                                                                                f1,
                                                                              loss.item()))
                    

            if (epoch + 1) % 1 == 0:
                self.validation()
            #self.test()
        torch.save(self.model.state_dict(), "/Users/atabeyunlu/ZubyNet-PyTorch/saved_model/cnn1_epoch_{}_lr_{}.pt".format(epoch,self.learning_rate))
                       
    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.float().to(self.device)
                labels = labels.float().to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
        

    def validation(self):
        self.model.eval()
        preds = []
        label_acc = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.float().to(self.device)
                
                outputs = self.model(images)
                preds += list(torch.argmax(outputs, dim=1).cpu().numpy())
                label_acc += list(labels)
            mcc = matthews_corrcoef(label_acc, preds)
            acc = accuracy_score(label_acc, preds)
            f1  = f1_score(label_acc, preds, average='macro')
            wandb.log({"val_accuracy": acc, "val_mcc": mcc, "val_f1_score": f1})
            print('Validation Accuracy: {:.4f}, MCC: {:.4f}, F1: {:.4f}'.format(acc,
                                                                                mcc,
                                                                                f1))
        self.model.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--bs', type=int, default=16, help='batch size')


    config = parser.parse_args()

    train = Train(config)

    train.training()
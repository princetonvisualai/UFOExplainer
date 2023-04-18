from __future__ import print_function
import torch
import os
import argparse
import pickle
import numpy as np
import scipy
import utils
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


class just_rotation(nn.Module):
    
    def __init__(self, in_dimension = 512, total_attr = 1200, device = torch.device('cuda')):
        
        super().__init__()
        self.rotation = nn.Linear(in_dimension, total_attr)
        self.require_all_grads()

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x, temp=None):
        return self.rotation(x)


if __name__=="__main__":
    
     
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100, metavar='epochs',
                        help='Epochs')
    parser.add_argument('--save', type=str, default='record/exp1', metavar='S',
                        help='save directory')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test on validation data')
    args = parser.parse_args()

    
    features, attr,  predictions365, logits, names = utils.get_ade20k_features()

    train_features = torch.Tensor(features['train'])
    val_features = torch.Tensor(features['val'])
    test_features = torch.Tensor(features['test'])

    train_attr = torch.Tensor(attr['train'])
    val_attr = torch.Tensor(attr['val'])
    test_attr = torch.Tensor(attr['test'])

    train_pred365 = torch.Tensor(predictions365['train'])
    val_pred365 = torch.Tensor(predictions365['val'])
    test_pred365 = torch.Tensor(predictions365['test'])

    train_logits = torch.Tensor(logits['train'])
    val_logits = torch.Tensor(logits['val'])
    test_logits = torch.Tensor(logits['test'])
    
    imp_attr = pickle.load(open('attr_splits/over20.pkl', 'rb'))


    

    model = just_rotation(in_dimension = train_features.shape[1],total_attr = len(imp_attr))

    train_attr = train_attr[:, imp_attr]
    val_attr = val_attr[:, imp_attr]
    test_attr = test_attr[:, imp_attr]
    print(train_attr.shape)
    
    model.train()
    
    device = torch.device('cuda')

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1, momentum=0.9)
    
    if args.faithful=='less':
        criterion1 = torch.nn.MSELoss()
        train_target = train_logits
    else:
        criterion1 = torch.nn.CrossEntropyLoss()
        train_target = train_pred365

    criterion2 = torch.nn.BCELoss()
    
    temp = 0.1
    batch = 512
    N = len(train_features)//batch + 1
    print(len(train_features)/batch) 
    #print(model.attr_select.weight.shape)
    expect_select = torch.zeros(16, train_attr.shape[1])
    expect_select[:, -1] = 1
    expect_select = expect_select.to(device)
    
    try:
        os.makedirs(args.save)
    except:
        pass
    
    if not args.test:
        for e in range(args.epochs):
            
            model.train()

            for t in range(N):
                feat_batch = train_features[t*batch:(t+1)*batch].to(device)
                attr_batch = train_attr[t*batch:(t+1)*batch].to(device)
                
                
                sc= model(feat_batch)

                crit = torch.nn.BCEWithLogitsLoss()
                optimizer.zero_grad()
                loss = crit(sc, attr_batch) 

                loss.backward()
                optimizer.step()
            

            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss}, 
                        '{}/feature_final.pth'.format(args.save)
                        )
            
            model.eval()
            sc = model(val_features.to(device))
            if e%2==0:
                to_print = []
                for i in range(sc.shape[1]):
                    if val_attr[:, i].sum()==0:
                        #to_print.append(-1)
                        continue
                    to_print.append(roc_auc_score(val_attr[:, i].squeeze(), sc[:, i].detach().cpu().numpy().squeeze()))
                print(e, loss, np.amin(to_print), flush=True)




    model.load_state_dict(torch.load('{}/feature_final.pth'.format(args.save))['model'])

    model.eval()
    sc = model(train_features.to(device))
    
    to_print = []
    for i in range(sc.shape[1]):
        if val_attr[:, i].sum()==0:
        to_print.append(roc_auc_score(train_attr[:, i].squeeze(), sc[:, i].detach().cpu().numpy().squeeze()))
    print(np.average(to_print), sep='\t' )
    print(to_print)

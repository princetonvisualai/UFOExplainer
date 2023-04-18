from __future__ import print_function
import torch
import os
import argparse
import pickle
import numpy as np
import scipy
import utils
import model
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time

def l1loss(mat):
    return torch.norm(mat, p=1)

if __name__=="__main__":
    
     
    parser = argparse.ArgumentParser()

    parser.add_argument('--faithful', type=str, default="somewhat") 
    parser.add_argument('--understandable', type=str, default="somewhat") 
    parser.add_argument('--num_attr', type=int, default=8, metavar='N',
                        help='number of attributes to use (default: 8)')
    parser.add_argument('--lmbda', type=float, default=0, metavar='reg',
                        help='learnability regularizer')
    parser.add_argument('--epochs', type=int, default=100, metavar='epochs',
                        help='Epochs')
    parser.add_argument('--outsize', type=str, default='group', metavar='out',
                        help='which predictions to use ')
    parser.add_argument('--scene', type=int, default=0, metavar='out',
                        help='which scene to compute')
    parser.add_argument('--save', type=str, default='record/exp1', metavar='S',
                        help='save directory')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test on validation data')

    args = parser.parse_args()

    
    if args.outsize=='all': 
        features, attr, predictions, logits, names = utils.get_ade20k_features(non_zero=True)
    elif args.outsize=='group':
        features, attr, predictions, logits, names = utils.get_ade20k_features_scenegroup()
    elif args.outsize=='binary':
        features, attr, predictions, logits, names = utils.get_ade20k_features_binary()

    train_features = torch.Tensor(features['train'])
    val_features = torch.Tensor(features['val'])
    test_features = torch.Tensor(features['test'])

    train_attr = torch.Tensor(attr['train'])
    val_attr = torch.Tensor(attr['val'])
    test_attr = torch.Tensor(attr['test'])
    print(train_attr.shape)

    train_pred365 = torch.Tensor(predictions['train'])
    val_pred365 = torch.Tensor(predictions['val'])
    test_pred365 = torch.Tensor(predictions['test'])
    
    train_logits = logits['train']
    val_logits = logits['val']
    test_logits = logits['test']
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    m_attr = model.just_rotation(in_dimension=train_features.shape[1], total_attr=train_attr.shape[1])
    m_attr.load_state_dict(torch.load('record/new_attr_align/feature_final.pth', map_location=device)['model'])
    m_attr.eval()
    

    train_rot_features = m_attr(train_features).detach()
    val_rot_features = m_attr(val_features).detach()
    test_rot_features = m_attr(test_features).detach()

    attr_losses_train = []
    relevant_attr = sorted([52, 45, 121, 203, 319, 215, 328, 175, 283, 182, 281, 51, 363, 307, 112, 256, 102, 352, 125, 26])
    
    
    train_target = torch.Tensor(train_logits)[:, relevant_attr[args.scene]].view(-1, 1)
    val_target = torch.Tensor(val_logits)[:, relevant_attr[args.scene]].view(-1, 1)
    test_target = torch.Tensor(test_logits)[:, relevant_attr[args.scene]].view(-1, 1)
    
    outsize = 1

    #to_use = []
    for a in range(train_rot_features.shape[1]):
        loss = torch.nn.BCEWithLogitsLoss()(train_rot_features[:, a], train_attr[:, a]).detach().data
        #print(a, loss, roc_auc_score(val_attr[:, a].detach().numpy(), val_rot_features[:, a].detach().numpy()))
        #if roc_auc_score(val_attr[:, a].detach().numpy(), val_rot_features[:, a].detach().numpy())>0.7:
        #    to_use.append(a)
        attr_losses_train.append(loss)
    #print(len(to_use))
    
    attr_losses_train = torch.Tensor(attr_losses_train).to(device)

    valid_attr = pickle.load(open('to_keep_somewhat_under.pkl', 'rb'))
    train_feat = torch.sigmoid(train_rot_features[:, valid_attr]).detach()
    #print(train_feat)
    val_feat = torch.sigmoid(val_rot_features[:, valid_attr]).detach()
    test_feat = torch.sigmoid(test_rot_features[:, valid_attr]).detach()
    
    print(train_target.shape, outsize, train_feat.shape)
    model = torch.nn.Linear(train_feat.shape[1], outsize) #model.less_faithful_less_understandable(n_outs=16, total_attr=len(loc_curr))
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    print(train_attr.shape)
    
    train_target = train_target.to(dtype=torch.float32)
    test_target = test_target.to(dtype=torch.float32)
    val_target = val_target.to(dtype=torch.float32)
    criterion1 = torch.nn.MSELoss()
        
    criterion2 = torch.nn.L1Loss()
    mses = []
    current = []
    #current = pickle.load(open('{}/chosen_attr.pkl'.format(args.save), 'rb'))
    
    try:
        os.makedirs(args.save)
    except:
        pass
     
    #expect_select = torch.zeros(args.num_attr, train_attr.shape[1])
    expect_select = torch.zeros(train_attr.shape[1])
    expect_select[-args.num_attr:] = 1

    expect_select = expect_select.to(device)
    min_loss = 1000000000000000
    #print(expect_select)

    #expect_unique = torch.zeros(train_attr.shape[1])
    #expect_unique[:args.num_attr] = 1
    #expect_unique = expect_unique.to(device)
    try:
        os.makedirs(args.save)
    except:
        pass
    
    temp = 0.1
    batch = 2048
    N = len(train_features)//batch + 1
    learn_attr = True
    temp = 0.005
    train_sum = train_feat.sum(dim=0)
    diffs = train_feat.max(dim=0)[0] - train_feat.min(dim=0)[0]
    diffs = diffs.to(device, dtype=torch.float32)
    diffs = torch.where(diffs ==0, torch.ones_like(diffs), diffs)

    if not args.test:
        for e in range(args.epochs):
            
            model.train()
            if e>=500 and e%100==0 and temp < 40:
               temp *= 2
            
            overall_loss = 0
            for t in range(N):
                feat_batch = train_feat[t*batch:(t+1)*batch].to(device)
                target_batch = train_target[t*batch:(t+1)*batch].to(device)
                
                #print(feat_batch.shape, attr_batch.shape, logits_batch.shape)
                sc = model(feat_batch)
                #exp_pred, attr_pred, attr_chosen = model(feat_batch, attr_batch)

                #print(exp_pred.shape, exp_pred) 
                #print(attr_pred.shape, attr_pred) 
                optimizer.zero_grad()
                loss = criterion1(sc, target_batch) #+ args.lmbda*(criterion2(attr_pred, attr_chosen.detach())) 
                if learn_attr:
                    #weight_sort = torch.sort(torch.sum(torch.abs(model.weight), dim=0))
                    #weight_order = weight_sort[0]
                    #weights_sum = torch.sum(model.weight*model.weight, dim=0)
                    weights_sum = torch.sum(model.weight*model.weight, dim=0)
                    weights_sum/=diffs
                    weight_sort = torch.sort(weights_sum)

                    loss_attr_selection = l1loss(weight_sort[0][:-args.num_attr].to(device)) #criterion2(weights_sum, torch.zeros_like(weights_sum).to(device)) 
                    loss+=temp*(loss_attr_selection)
                
                loss.backward()
                optimizer.step()
                
                #w = model.weight.data.clamp(min=0)
                #model.weight.data = w

                overall_loss+=loss
                
                #if loss_attr_selection<1e-5:
                #    learn_attr = False
                #    model.attr_select.weight = torch.nn.Parameter(torch.round(model.attr_select.weight))
                #    for param in model.attr_select.parameters():
                #        param.requires_grad = False
                #    loss_attr_selection = 0
            
            model.eval()
            
            with torch.no_grad():
                
                val_sc = model(val_feat.to(device))

                loss = criterion1(val_sc, val_target.to(device))
                
                weights_sum = torch.sum(model.weight*model.weight, dim=0)
                weight_sort = torch.sort(weights_sum)
                loss_attr_selection = l1loss(weight_sort[0][:-args.num_attr].to(device)) #criterion2(weights_sum, torch.zeros_like(weights_sum).to(device)) 
                
                curr_loss = loss+temp*(loss_attr_selection)
           
            #if e==499:
            #    temp = 0.01
            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':min_loss}, 
                        '{}/feature_current{}.pth'.format(args.save, args.scene)
                        )
            if e%20==0:
                print(e, temp, loss, loss_attr_selection, flush=True)
            
     
    
    import pandas as pd
    attr_names = pd.read_csv('../NetDissect-Lite/dataset/broden1_224/label.csv', index_col=0)['name'].to_dict()

    model.load_state_dict(torch.load('{}/feature_current{}.pth'.format(args.save, args.scene))['model'])
    

    weights_sum = torch.sum(model.weight*model.weight, dim=0)
    weights_sum/=diffs
    weight_sort = torch.sort(weights_sum)

    print(weight_sort)

    imp_attr = weight_sort[1][-args.num_attr:].detach().cpu().numpy()
    
    pickle.dump(imp_attr, open('{}/imp_attr{}.pkl'.format(args.save, args.scene), 'wb+'))
    
    over20 = pickle.load(open('over20.pkl', 'rb'))

    for i, a in enumerate(imp_attr):
        print(attr_names[over20[valid_attr[a]]]) 


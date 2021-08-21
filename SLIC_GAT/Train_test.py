import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
import numpy as np
import copy

import os
import sys
import pathlib
pth=str(pathlib.Path().absolute())

#Utilities
sys.path.append(pth)
from model_An import *
from util import *
from Data_loader_image import *

#Base de datos
#sys.path.append()
#data=("/").join(pth.split("/")[:-2])+"/Data_Base/Metada_V6G_p1"
data=("/").join(pth.split("/")[:-2])+"/Data_Base/Metada_V6G"
data_arg=("/").join(pth.split("/")[:-2])+"/Data_Base"

to_cuda = to_cuda

def train_model(
    dataset,
        epochs,
        batch_size,
        use_cuda,
    folds=5,
        disable_tqdm=False,
        ):
    print("Reading dataset")

    ind=np.arange(0,len(dataset))
    
    #INSIDE K-FOLD
    results={}
    for fold in range(folds):
        #generate train-test
        print("fold"+str(fold))
        
        i=fold

        ind=np.arange(0,len(dataset))
        indexes=ind[np.random.permutation(len(dataset))]
        L=len(dataset)
        test_idx=indexes[int(i*L/5):int((i+1)*L/5)]
        train_idx=np.delete(indexes,np.arange(int(i*L/5),int((i+1)*L/5)))
        
        model = GAT_ANE_(41,1)
        if use_cuda:
            model = model.cuda()
    
        opt = torch.optim.Adam(model.parameters())
    
        best_valid_acc = 0.
        best_model = copy.deepcopy(model)
    
        last_epoch_train_loss = 0.
        last_epoch_train_acc = 0.
        last_epoch_valid_acc = 0.
    
        interrupted = False
        
        train_dat=np.vectorize(lambda ind:dataset[ind],otypes=[object])(train_idx)
        train_graph=np.vectorize(lambda b:b["image_graph"])(train_dat)
        train_label=np.vectorize(lambda b:b["landmarks"])(train_dat)
        test_dat=np.vectorize(lambda ind:dataset[ind],otypes=[object])(test_idx)
        test_graph=np.vectorize(lambda b:b["image_graph"])(test_dat)
        test_label=np.vectorize(lambda b:b["landmarks"])(test_dat)
        
        loss_function = nn.MSELoss()
        for e in tqdm(range(epochs), total=epochs, desc="Epoch ", disable=disable_tqdm,):
            try:
                #train_losses, train_accs = train_(model, opt, train_graph, train_label,loss_function,
                                                  #batch_size=batch_size, use_cuda=use_cuda, disable_tqdm=disable_tqdm,)
                train_losses, train_accs =train_(model=model,
                                                 optimiser=opt,
                                                 graphs=train_graph,
                                                 labels=train_label,
                                                 use_cuda=use_cuda,
                                                 loss_function=nn.MSELoss(),
                                                 #batch_size=1,
                                                 batch_size=batch_size,
                                                 disable_tqdm=disable_tqdm,
                                                 profile=False)
            
                last_epoch_train_loss = np.mean(train_losses)
                last_epoch_train_acc = 100*np.mean(train_accs)
            except KeyboardInterrupt:
                print("Training interrupted!")
                interrupted = True
        
            #valid_accs = test_(model,test_graph,test_label,use_cuda,desc="Validation ", disable_tqdm=disable_tqdm,)
            valid_accs = test_(model=model,
                               graphs=test_graph,
                               labels=test_label,
                               use_cuda=use_cuda,
                               batch_size=int(32),
                               desc="Test ",
                               disable_tqdm=False)
                
            last_epoch_valid_acc = 100*np.mean(valid_accs)
        
            if last_epoch_valid_acc>best_valid_acc:
                best_valid_acc = last_epoch_valid_acc
                best_model = copy.deepcopy(model)
        
            tqdm.write("EPOCH SUMMARY {loss:.4f} {t_acc:.2f}% {v_acc:.2f}%".format(loss=last_epoch_train_loss, t_acc=last_epoch_train_acc, v_acc=last_epoch_valid_acc))
        
            if interrupted:
                break
    
        results[fold]={"train_acc":train_accs,"train_loss":train_losses,"valid_acc":valid_accs}
        save_model("best"+str(fold),best_model)
        save_model("last"+str(fold),model)
    np.save("results"+'.npy',results)
    return results



def main(
        train:bool=True,
        test:bool=False,
        epochs:int=100,
        batch_size:int=32,
        use_cuda:bool=True,
        disable_tqdm:bool=False,
        Data_version = "Metadata_V6G",
        data_arg=data_arg
        ):
    use_cuda = use_cuda and torch.cuda.is_available()

    dataset=Rotated_Dataset(data_arg,Data_version)

    if train:

        results=train_model(dataset,
                epochs=35,
                batch_size=int(100),
                use_cuda=True,
                folds=5,
                disable_tqdm=False,
                )
    #if test:
    #    test_model(
    #            use_cuda=use_cuda,
    #            dset_folder = dset_folder,
    #            disable_tqdm = disable_tqdm,
    #            )

if __name__ == "__main__":
    fire.Fire(main)

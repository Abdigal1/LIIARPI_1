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
data=("/").join(pth.split("/")[:-2])+"/lab/Data_Base/Metada_V6G"
data_arg=("/").join(pth.split("/")[:-2])+"/lab/Data_Base"

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

    sub_dir="MMhh_Sel_L2/"
    print(os.path.join(os.path.dirname(os.path.realpath(__file__)),sub_dir))

    ind=np.arange(0,len(dataset))
    
    #INSIDE K-FOLD
    last_results={}
    results={}
    results_abs={}
    for fold in range(folds):
        #generate train-test
        print("fold "+str(fold+1))
        
        i=fold

        ind=np.arange(0,len(dataset))
        indexes=ind[np.random.permutation(len(dataset))]
        L=len(dataset)
        test_idx=indexes[int(i*L/folds):int((i+1)*L/folds)]
        train_idx=np.delete(indexes,np.arange(int(i*L/folds),int((i+1)*L/folds)))
        
        model = GAT_ANE_MHH(9,1)
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
        
        loss_function = nn.L1Loss()
        
        epoch_train=[]
        epoch_train_abs=[]
        epoch_test=[]
        epoch_test_abs=[]
        for e in tqdm(range(epochs), total=epochs, desc="Epoch ", disable=disable_tqdm,):
            try:
                #train_losses, train_accs = train_(model, opt, train_graph, train_label,loss_function,
                                                  #batch_size=batch_size, use_cuda=use_cuda, disable_tqdm=disable_tqdm,)
                train_losses, train_accs,train_accs_abs =train_(model=model,
                                                 optimiser=opt,
                                                 graphs=train_graph,
                                                 labels=train_label,
                                                 use_cuda=use_cuda,
                                                 loss_function=nn.MSELoss(),
                                                 #loss_function=nn.L1Loss(),
                                                 #batch_size=1,
                                                 batch_size=batch_size,
                                                 disable_tqdm=disable_tqdm,
                                                 profile=False)
            
                last_epoch_train_loss = np.mean(train_losses)
                last_epoch_train_acc = np.mean(train_accs)
                last_epoch_train_loss_l1 = np.mean(train_accs_abs)

            except KeyboardInterrupt:
                print("Training interrupted!")
                interrupted = True
        
            #valid_accs = test_(model,test_graph,test_label,use_cuda,desc="Validation ", disable_tqdm=disable_tqdm,)
            valid_accs,loss_l1 = test_(model=model,
                               graphs=test_graph,
                               labels=test_label,
                               use_cuda=use_cuda,
                               batch_size=batch_size,
                               desc="Test ",
                               disable_tqdm=False)
                
            last_epoch_valid_acc = np.mean(valid_accs)
            last_epoch_valid_acc_l1 = np.mean(loss_l1)
        
            if last_epoch_valid_acc>best_valid_acc:
                best_valid_acc = last_epoch_valid_acc
                best_model = copy.deepcopy(model)
        
            epoch_train.append(last_epoch_train_loss)
            epoch_train_abs.append(last_epoch_train_loss_l1)
            epoch_test.append(last_epoch_valid_acc)
            epoch_test_abs.append(last_epoch_valid_acc_l1)
            tqdm.write("EPOCH SUMMARY TRAIN [ L2: {train_loss_l2:.4f} L1:  {train_loss_l1:.2f} ] TEST [ L2: {test_loss_l2:.4f} L1:  {test_loss_l1:.2f} ]".format(
                train_loss_l2=last_epoch_train_loss,
                train_loss_l1=last_epoch_train_loss_l1,
                test_loss_l2=last_epoch_valid_acc,
                test_loss_l1=last_epoch_valid_acc_l1,
                ))
        
            if interrupted:
                break
    
        last_results[fold]={"train_acc":train_accs,
                            "train_loss":epoch_train,
                            "valid_acc":epoch_test
                            }

        results[fold]={"train_acc":last_epoch_train_acc,
                        "train_loss":last_epoch_train_loss,
                        "valid_acc":last_epoch_valid_acc
                        }

        results_abs[fold]={"train_acc_abs":epoch_train_abs,
                        "valid_acc_abs":epoch_test_abs
                        }

        save_model(os.path.join(os.path.dirname(os.path.realpath(__file__)),sub_dir)+"best"+str(fold),best_model)
        save_model(os.path.join(os.path.dirname(os.path.realpath(__file__)),sub_dir)+"last"+str(fold),model)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),sub_dir)+"results_10f_unsampled_hh"+'.npy',results)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),sub_dir)+"last_results_10f_unsampled_hh"+'.npy',last_results)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),sub_dir)+"last_results_abs_10f_unsampled_hh"+'.npy',results_abs)
    return results,last_results,results_abs



def main(
        train:bool=True,
        test:bool=False,
        epochs:int=100,
        batch_size:int=300,
        use_cuda:bool=True,
        disable_tqdm:bool=False,
        Data_version = "Metadata_V7G_pytorch",
        data_arg=data_arg
        ):
    use_cuda = use_cuda and torch.cuda.is_available()

    #dataset=Rotated_Dataset(data_arg,Data_version)
    dataset=Rotated_Dataset(data_arg,Data_version,
                        features=[
                            12,
                            13,
                            15,
                            16,
                            -5,
                            -4,
                            -3,
                            -2,
                            -1,
                        ]
                       )

    if train:

        results=train_model(dataset,
                epochs=int(250),
                batch_size=int(350),
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

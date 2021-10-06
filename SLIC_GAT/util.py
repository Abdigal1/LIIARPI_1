from tqdm import tqdm
import fire

import time
import pickle
import multiprocessing

import numpy as np
import scipy as sp
from skimage.segmentation import slic, mark_boundaries
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST

from model import GAT_MNIST

NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

NUM_FEATURES = 3
NUM_CLASSES = 10

def plot_image(image,desired_nodes=75,save_in=None):
    # show the output of SLIC
    fig = plt.figure("Image")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)#, cmap="gray")
    plt.axis("off")
    
    # show the plots
    if save_in is None:
        plt.show()
    else:
        plt.savefig(save_in,bbox_inches="tight")
    plt.close()

def plot_graph_from_image(image,desired_nodes=75,save_in=None):
    segments = slic(image, slic_zero = True)

    # show the output of SLIC
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(mark_boundaries(image, segments), cmap="gray")
    ax.imshow(image)#, cmap="gray")
    plt.axis("off")

    asegments = np.array(segments)

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments

    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    plt.scatter(centers[:,1],centers[:,0], c='r')

    for i in range(bneighbors.shape[1]):
        y0,x0 = centers[bneighbors[0,i]]
        y1,x1 = centers[bneighbors[1,i]]

        l = Line2D([x0,x1],[y0,y1], c="r", alpha=0.5)
        ax.add_line(l)

    # show the plots
    if save_in is None:
        plt.show()
    else:
        plt.savefig(save_in,bbox_inches="tight")
    plt.close()

def plot_graph_from_image(image,desired_nodes=75,save_in=None):
    segments = slic(image, slic_zero = True)

    # show the output of SLIC
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(mark_boundaries(image, segments), cmap="gray")
    ax.imshow(image)#, cmap="gray")
    plt.axis("off")

    asegments = np.array(segments)

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments

    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    plt.scatter(centers[:,1],centers[:,0], c='r')

    for i in range(bneighbors.shape[1]):
        y0,x0 = centers[bneighbors[0,i]]
        y1,x1 = centers[bneighbors[1,i]]

        l = Line2D([x0,x1],[y0,y1], c="r", alpha=0.5)
        ax.add_line(l)

    # show the plots
    if save_in is None:
        plt.show()
    else:
        plt.savefig(save_in,bbox_inches="tight")
    plt.close()

def get_graph_from_image(image,desired_nodes=75):
    # load the image and convert it to a floating point data type
    segments = slic(image, n_segments=desired_nodes, slic_zero = True)
    asegments = np.array(segments)

    num_nodes = np.max(asegments)
    nodes = {
        node: {
            "rgb_list": [],
            "pos_list": []
        } for node in range(num_nodes+1)
    }

    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            node = asegments[y,x]
            rgb = image[y,x,:]
            pos = np.array([float(x)/width,float(y)/height])
            nodes[node]["rgb_list"].append(rgb)
            nodes[node]["pos_list"].append(pos)
        #end for
    #end for
    
    G = nx.Graph()
    
    for node in nodes:
        nodes[node]["rgb_list"] = np.stack(nodes[node]["rgb_list"])
        nodes[node]["pos_list"] = np.stack(nodes[node]["pos_list"])
        # rgb
        rgb_mean = np.mean(nodes[node]["rgb_list"], axis=0)
        #rgb_std = np.std(nodes[node]["rgb_list"], axis=0)
        #rgb_gram = np.matmul( nodes[node]["rgb_list"].T, nodes[node]["rgb_list"] ) / nodes[node]["rgb_list"].shape[0]
        # Pos
        pos_mean = np.mean(nodes[node]["pos_list"], axis=0)
        #pos_std = np.std(nodes[node]["pos_list"], axis=0)
        #pos_gram = np.matmul( nodes[node]["pos_list"].T, nodes[node]["pos_list"] ) / nodes[node]["pos_list"].shape[0]
        # Debug
        
        features = np.concatenate(
          [
            np.reshape(rgb_mean, -1),
            #np.reshape(rgb_std, -1),
            #np.reshape(rgb_gram, -1),
            np.reshape(pos_mean, -1),
            #np.reshape(pos_std, -1),
            #np.reshape(pos_gram, -1)
          ]
        )
        G.add_node(node, features = list(features))
    #end
    
    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:,:-1].ravel(), segments[:,1:].ravel()])
    vs_below = np.vstack([segments[:-1,:].ravel(), segments[1:,:].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    # Adjacency loops
    for i in range(bneighbors.shape[1]):
        if bneighbors[0,i] != bneighbors[1,i]:
            G.add_edge(bneighbors[0,i],bneighbors[1,i])
    
    # Self loops
    for node in nodes:
        G.add_edge(node,node)
    
    n = len(G.nodes)
    m = len(G.edges)
    h = np.zeros([n,NUM_FEATURES]).astype(NP_TORCH_FLOAT_DTYPE)
    edges = np.zeros([2*m,2]).astype(NP_TORCH_LONG_DTYPE)
    for e,(s,t) in enumerate(G.edges):
        edges[e,0] = s
        edges[e,1] = t
        
        edges[m+e,0] = t
        edges[m+e,1] = s
    #end for
    for i in G.nodes:
        h[i,:] = G.nodes[i]["features"]
    #end for
    del G
    return h, edges

def batch_graphs(gs):
    NUM_FEATURES = gs[0][0].shape[-1]
    G = len(gs)
    N = sum(g[0].shape[0] for g in gs) #Sum of nodes n*#img
    M = sum(g[1].shape[0] for g in gs) #Sum of edges 2m*#img
    adj = np.zeros([N,N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N,M])
    Mtgt = np.zeros([N,M])
    Mgraph = np.zeros([N,G])
    h = np.concatenate([g[0] for g in gs])
    
    n_acc = 0
    m_acc = 0
    for g_idx, g in enumerate(gs):
        n = g[0].shape[0]
        m = g[1].shape[0]
        
        for e,(s,t) in enumerate(g[1]):
            adj[int(n_acc+s),int(n_acc+t)] = 1
            adj[int(n_acc+t),int(n_acc+s)] = 1
            
            src[int(m_acc+e)] = n_acc+s
            tgt[int(m_acc+e)] = n_acc+t
            
            Msrc[int(n_acc+s),int(m_acc+e)] = 1
            Mtgt[int(n_acc+t),int(m_acc+e)] = 1
            
        Mgraph[int(n_acc):int(n_acc+n),int(g_idx)] = 1
        
        n_acc += n
        m_acc += m
    return (
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE)
    )

def batch_graphs_(gs):
    NUM_FEATURES = gs[0][0].shape[-1]
    G = 1
    N = sum(g[0].shape[0] for g in gs) #Sum of nodes n*#img
    M = sum(g[1].shape[0] for g in gs) #Sum of edges 2m*#img
    N=gs[0].shape[0]
    M=gs[1].shape[0]
    print(N)
    print(M)
    adj = np.zeros([N,N])
    src = np.zeros([M])
    tgt = np.zeros([M])
    Msrc = np.zeros([N,M])
    Mtgt = np.zeros([N,M])
    Mgraph = np.zeros([N,G])
    h = np.concatenate([g[0] for g in gs])
    
    n_acc = 0
    m_acc = 0
    #for g_idx, g in enumerate(gs):
    n = g[0].shape[0]
    m = g[1].shape[0]
    #print(gs[1])
    print(src.shape)
        
    for e,(s,t) in enumerate(gs[1]):
        adj[int(n_acc+s),int(n_acc+t)] = 1
        adj[int(n_acc+t),int(n_acc+s)] = 1
        
        print(int(m_acc+e))
        src[int(m_acc+e)] = n_acc+s
        tgt[int(m_acc+e)] = n_acc+t
            
        Msrc[int(n_acc+s),int(m_acc+e)] = 1
        Mtgt[int(n_acc+t),int(m_acc+e)] = 1
            
    Mgraph[int(n_acc):int(n_acc+n),0] = 1
        
    #n_acc += n
    #m_acc += m
    return (
        h.astype(NP_TORCH_FLOAT_DTYPE),
        adj.astype(NP_TORCH_FLOAT_DTYPE),
        src.astype(NP_TORCH_LONG_DTYPE),
        tgt.astype(NP_TORCH_LONG_DTYPE),
        Msrc.astype(NP_TORCH_FLOAT_DTYPE),
        Mtgt.astype(NP_TORCH_FLOAT_DTYPE),
        Mgraph.astype(NP_TORCH_FLOAT_DTYPE)
    )
    
def save_model(fname, model):
    torch.save(model.state_dict(),"{fname}.pt".format(fname=fname))
    
def load_model(fname, model):
    model.load_state_dict(torch.load("{fname}.pt".format(fname=fname)))

def to_cuda(x):
    return x.cuda()
    
def split_dataset(labels, valid_split=0.1):
    idx = np.random.permutation(len(labels))
    valid_idx = []
    train_idx = []
    label_count = [0 for _ in range(1+max(labels))]
    valid_count = [0 for _ in label_count]
    
    for i in idx:
        label_count[labels[i]] += 1
    
    
    for i in idx:
        l = labels[i]
        if valid_count[l] < label_count[l]*valid_split:
            valid_count[l] += 1
            valid_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, valid_idx

def con_mat(y,yp,ranges):
  pr1r=np.vectorize(lambda y,range:np.logical_and(range[0]<=y,y<range[1]),signature="(),(i)->()")(yp,ranges)
  pr2r=np.vectorize(lambda y,range:np.logical_and(range[0]<=y,y<range[1]),signature="(),(i)->()")(y,ranges)
  TP=np.sum(pr2r*pr1r,axis=0)
  TN=np.sum(np.logical_not(pr2r)*np.logical_not(pr1r),axis=0)
  FN=np.sum((pr2r)*np.logical_not(pr1r),axis=0)
  FP=np.sum(np.logical_not(pr2r)*(pr1r),axis=0)
  return TP,TN,FP,FN

def train(model, optimiser, graphs, labels, train_idx, use_cuda, batch_size=1, disable_tqdm=False, profile=False):
    train_losses = []
    train_accs = []
    
    indexes = train_idx[np.random.permutation(len(train_idx))]
    pyt_labels = torch.tensor(labels)
    
    if use_cuda:
        pyt_labels = pyt_labels.cuda()
    
    for b in tqdm(range(0,len(indexes),batch_size), total=len(indexes)/batch_size, desc="Instances ", disable=disable_tqdm):
        ta = time.time()
        optimiser.zero_grad()
        
        batch_indexes = indexes[b:b+batch_size]
        
        batch_labels = pyt_labels[batch_indexes]
        tb = time.time()
        h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs(graphs[batch_indexes])
        tc = time.time()
        h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.from_numpy,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
        td = time.time()
        if use_cuda:
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
        te = time.time()
        y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
        tf = time.time()
        y.shape
        batch_labels.shape
        #loss = F.cross_entropy(input=y,target=batch_labels)
        loss = F.l1_loss(input=y,target=batch_labels)
        
        pred = torch.argmax(y,dim=1).detach().cpu().numpy()
        acc = np.sum((pred==labels[batch_indexes]).astype(float)) / batch_labels.shape[0]
        mode = sp.stats.mode(pred)
        tg = time.time()
        
        tqdm.write(
              "{loss:.4f}\t{acc:.2f}%\t{mode} (x{modecount})".format(
                  loss=loss.item(),
                  acc=100*acc,
                  mode=mode[0][0],
                  modecount=mode[1][0],
              )
        )
        
        th = time.time()
        loss.backward()
        optimiser.step()
        
        train_losses.append(loss.detach().cpu().item())
        train_accs.append(acc)
        if profile:
            ti = time.time()
            
            tt = ti-ta
            tqdm.write("zg {zg:.2f}% bg {bg:.2f}% tt {tt:.2f}% tc {tc:.2f}% mo {mo:.2f}% me {me:.2f}% bk {bk:.2f}%".format(
                    zg=100*(tb-ta)/tt,
                    bg=100*(tc-tb)/tt,
                    tt=100*(td-tc)/tt,
                    tc=100*(te-td)/tt,
                    mo=100*(tf-te)/tt,
                    me=100*(tg-tf)/tt,
                    bk=100*(ti-th)/tt,
                    ))
        
    return train_losses, train_accs

def train_(
    model,
    optimiser,
    graphs,
    labels,
    use_cuda,
    loss_function=nn.MSELoss(),
    batch_size=1,
    disable_tqdm=False,
    profile=False):

    train_losses = []
    train_accs = []
    train_accs_abs = []
    train_sensitivity=[]
    train_specificity=[]
    train_TP=[]
    train_TN=[]
    train_FP=[]
    train_FN=[]

    #loss_function = nn.MSELoss()
    #indexes = train_idx[np.random.permutation(len(train_idx))]
    #labels=np.vectorize(lambda ind:dataset[ind],otypes=[object])(train_idx)
    pyt_labels = torch.tensor(labels.reshape(-1,1),dtype=torch.float32)
    
    if use_cuda:
        pyt_labels = pyt_labels.cuda()
    
    for b in tqdm(range(0,len(labels),batch_size), total=len(labels)/batch_size, desc="Instances ", disable=disable_tqdm):
        ta = time.time()
        optimiser.zero_grad()
        
        #batch_indexes = indexes[b:b+batch_size]
        
        batch_labels = pyt_labels[b:b+batch_size]
        tb = time.time()
        h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs(graphs[b:b+batch_size])
        tc = time.time()
        h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.from_numpy,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
        td = time.time()
        if use_cuda:
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
        te = time.time()
        y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
        tf = time.time()
        

        loss = torch.tensor([0], dtype=torch.float).cuda()
        #loss = F.l1_loss(input=y,target=batch_labels)
        #loss = F.mse_loss(input=y.cuda(),target=batch_labels.cuda())
        #loss = F.nll_loss(input=y.cuda(),target=batch_labels.cuda())

        #loss_function = nn.MSELoss()
        loss = loss_function(y, batch_labels)
        
        
        pred=y.detach().cpu().numpy()
#
        #
        acc=np.sum((pred-batch_labels.cpu().numpy())**2) / batch_labels.shape[0]

        acc_l1=np.sum(np.abs(pred-batch_labels.cpu().numpy())) / batch_labels.shape[0]

        TP,TN,FP,FN=con_mat(batch_labels.cpu().numpy(),pred,
                            np.array([[0,6],[6,8],[8,10],[10,13],[13,25]])
                            #np.array([[0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[14,25]])
                            )
        
        #print(TP)
        #print(FN)
        sen=TP/(TP+FN)
        spe=TN/(FP+TN)
        clacc=(TP+TN)/batch_labels.cpu().numpy().shape[0]

        mode = sp.stats.mode(pred)
        tg = time.time()
        
        tqdm.write(
              "{loss:.2f}\t{acc:.2f}\t {sen} {spe} {clacc} {mode} (x{modecount})".format(
                  loss=acc_l1,
                  acc=acc,
                  sen=sen,
                  spe=spe,
                  clacc=clacc,
                  mode=mode[0][0],
                  modecount=mode[1][0],
              )
        )
        
        th = time.time()
        loss.backward()
        optimiser.step()
        
        train_losses.append(loss.detach().cpu().item())
        train_accs.append(acc)
        train_accs_abs.append(acc_l1)
        train_sensitivity.append(sen)
        train_specificity.append(spe)
        train_TP.append(TP)
        train_TN.append(TN)
        train_FP.append(FP)
        train_FN.append(FN)

        if profile:
            ti = time.time()
            
            tt = ti-ta
            tqdm.write("zg {zg:.2f}% bg {bg:.2f}% tt {tt:.2f}% tc {tc:.2f}% mo {mo:.2f}% me {me:.2f}% bk {bk:.2f}%".format(
                    zg=100*(tb-ta)/tt,
                    bg=100*(tc-tb)/tt,
                    tt=100*(td-tc)/tt,
                    tc=100*(te-td)/tt,
                    mo=100*(tf-te)/tt,
                    me=100*(tg-tf)/tt,
                    bk=100*(ti-th)/tt,
                    ))
        
    return train_losses, train_accs,train_accs_abs,train_sensitivity,train_specificity,train_TP,train_TN,train_FP,train_FN

def test(model, graphs, labels, use_cuda, desc="Test ", disable_tqdm=False):
    test_accs = []
    for i in tqdm(range(len(indexes)), total=len(indexes), desc=desc, disable=disable_tqdm):
        with torch.no_grad():
            idx = indexes[i]
        
            batch_labels = labels[idx:idx+1]
            pyt_labels = torch.from_numpy(batch_labels)
            
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs([graphs[idx]])
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.from_numpy,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
            
            if use_cuda:
                h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels))
            
            y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
            
            pred = torch.argmax(y,dim=1).detach().cpu().numpy()
            acc = np.sum((pred==batch_labels).astype(float)) / batch_labels.shape[0]
            
            test_accs.append(acc)
    return test_accs

def test_(model,
            graphs,
             labels,
             use_cuda,
             desc="Test ",
             batch_size=1,
             disable_tqdm=False):
    test_accs = []
    test_accs_l1 = []
    test_sensitivity=[]
    test_specificity=[]
    test_TP=[]
    test_TN=[]
    test_FP=[]
    test_FN=[]
    #for i in tqdm(range(len(labels)), total=len(labels), desc=desc, disable=disable_tqdm):
    for b in tqdm(range(0,len(labels),batch_size), total=len(labels)/batch_size, desc=desc, disable=disable_tqdm):
        with torch.no_grad():
            #idx = indexes[i]
        
            #batch_labels = labels[idx:idx+1]
            batch_labels = labels
            #pyt_labels = torch.from_numpy(batch_labels,dtype=torch.long)
            pyt_labels = torch.tensor(labels.reshape(-1,1),dtype=torch.float32)
            pyt_labels=pyt_labels[b:b+batch_size]
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = batch_graphs(graphs[b:b+batch_size])
            h,adj,src,tgt,Msrc,Mtgt,Mgraph = map(torch.from_numpy,(h,adj,src,tgt,Msrc,Mtgt,Mgraph))
            
            if use_cuda:
                h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels = map(to_cuda,(h,adj,src,tgt,Msrc,Mtgt,Mgraph,pyt_labels))
            
            y = model(h,adj,src,tgt,Msrc,Mtgt,Mgraph)
            pred=y.detach().cpu().numpy()

            acc=np.sum((pred-pyt_labels.cpu().numpy())**2) / pyt_labels.cpu().numpy().shape[0]

            acc_l1=np.sum(np.abs(pred-pyt_labels.cpu().numpy())) / pyt_labels.cpu().numpy().shape[0]
           
            TP,TN,FP,FN=con_mat(pyt_labels.cpu().numpy(),pred,
                            np.array([[0,6],[6,8],[8,10],[10,13],[13,25]])
                            #np.array([[0,2],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14],[14,25]])
                                        )

            sen=TP/(TP+FN)
            spe=TN/(FP+TN)
            clacc=(TP+TN)/pyt_labels.cpu().numpy().shape[0]
            
            test_accs.append(acc)
            test_accs_l1.append(acc_l1)
            test_sensitivity.append(sen)
            test_specificity.append(spe)
            test_TP.append(TP)
            test_TN.append(TN)
            test_FP.append(FP)
            test_FN.append(FN)
            #test_accs.append(loss.detach().cpu().item())
    return test_accs,test_accs_l1,test_sensitivity,test_specificity,test_TP,test_TN,test_FP,test_FN

def main_plot(dset_folder,save):
    dset = MNIST(dset_folder,download=True)
    imgs = dset.data.unsqueeze(-1).numpy().astype(np.float64)
    labels = dset.targets.numpy()
    total_labels = set(labels)
    plotted_labels = set()
    i = 0
    while len(plotted_labels) < len(total_labels):
        if labels[i] not in plotted_labels:
            plotted_labels.add(labels[i])
            plot_image(imgs[i,:,:,0],save_in=(None if not save else "{}i.png".format(labels[i])))
            plot_graph_from_image(imgs[i,:,:,0],save_in=(None if not save else "{}g.png".format(labels[i])))
        i+=1

def main(
        plot_mnist:bool=False,
        save_plot_mnist:bool=False,
        dset_folder:str = "./mnist"
        ):
    if plot_mnist or save_plot_mnist:
        main_plot(dset_folder=dset_folder,save=save_plot_mnist)


if __name__ == "__main__":
    fire.Fire(main)

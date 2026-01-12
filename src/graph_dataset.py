from dgl.data.utils import load_graphs
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import json
from dgl.data import DGLDataset

def create_dataset(dataset,data_folder):
  labels=[]
  missed=[]
  graphs=[]
  names=[]
  for _,row in tqdm(dataset.iterrows()):
        file = row["File"]
        graph_of_interest=f'{data_folder}/{file}-fact/_graphs/{row["pair"]}.bin'

        if(os.path.exists(graph_of_interest)):
          try:
            graph , _= load_graphs(graph_of_interest)
            graphs.append(graph[0])
            labels.append([row['overlap GT'],row['preproc GT']])
            names.append(f'{file}/{row["pair"]}')
          except:
              continue
        else:
          missed.append(graph_of_interest)
  if(len(labels)==0):
      raise Exception("The specified folder do not contain the training data specified in the csv file")
  labels=torch.from_numpy(np.stack(labels,axis=0)).to(torch.float32)
  print(f"{len(missed)} model pairs could not be found")
  return graphs,labels,missed,names


class Dataset(DGLDataset):
    _url = ''
    _sha1_str = ''
    def __init__(self, dataset_path ,data_path,raw_dir=None,force_reload=False, verbose=False):
        self.dataset_path=dataset_path
        self.data_path=data_path
        super(Dataset, self).__init__(name='Data Leakage dataset',
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)


    def process(self):
        # process data to a list of graphs and a list of labels
        self.graphs, self.label,self.target_names = self._load_graph()
        

    def _load_graph(self):
        if self.data_path.endswith(".bin"):
            print("Loading the preprocessed graphs")
            graphs, labels = load_graphs(self.data_path)
            print(f"Num of loaded graphs {len(graphs)}")
            labels = labels["labels"]
            df = pd.read_csv(self.dataset_path, delimiter=",")
            target_names = df["File"].tolist()
            return graphs, labels, target_names

        dataset = pd.read_csv(self.dataset_path)
        graphs,labels,_,target_names=create_dataset(dataset,self.data_path)

        return graphs, labels, target_names

    @property
    def get_labels(self):
        return self.label

    @property
    def num_labels(self):
        return 2

    @property
    def feature_size(self):
        return self.graphs[0].ndata['features'].size()[1]

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx],self.target_names[idx]

    def __len__(self):
        return len(self.graphs)
    

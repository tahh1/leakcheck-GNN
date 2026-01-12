import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import re
import torch
import os
import dgl
from dgl.data.utils import save_graphs
from .utils import remove_files

INVALID_VAR_PATTERN = re.compile(r"\[(\$?invo\d+?)?, (\$?invo\d+?)?\]")


def extract_instr_and_vars(FlowVarTransformation,FlowVarStoreIndex):

    unique_instr = sorted(set(
                            read_index_or_empty(FlowVarTransformation,"InstructionId")
                            +read_index_or_empty(FlowVarStoreIndex,"InstructionId")))
    index_mapping = {instr:index for index,instr in enumerate(unique_instr)}
    unique_vars = sorted(set(
                           read_index_or_empty(FlowVarTransformation,"ToId")+
                           read_index_or_empty(FlowVarTransformation,"FromId")+
                           read_index_or_empty(FlowVarStoreIndex,"ToId")+
                           read_index_or_empty(FlowVarStoreIndex,"FromId")))
    unique_vars = list(filter(lambda x: INVALID_VAR_PATTERN.fullmatch(x)==None,unique_vars)) #Filter empty vars ([, ])
    num_instr= len(index_mapping)
    for index,var in enumerate(unique_vars):
        index_mapping[var]=num_instr+index

    return unique_instr,unique_vars,index_mapping


def preprocess_flow_df(df):
     df["To"]=df["To"].fillna("")
     df["From"]=df["From"].fillna("")
     df["InstructionId"] = df["ToCtx"] + df["Instr"].astype(str) + df["FromCtx"]
     df["ToId"] = df["To"] + df["ToCtx"]
     df["FromId"] = df["From"] + df["FromCtx"]
     return df


def process_flow_df(df, lines,
                    instr_labels, instr_meths, instr_loc,
                    flow_from_inst, flow_to_inst,
                    var_loc, var_labels):
    instr_number_pattern = r'(?<=\])\d+'
    for _, row in df.iterrows():
        instr_id = row["InstructionId"]
        to_id = row["ToId"]
        from_id = row["FromId"]

        if instr_id not in instr_labels:
            instr_labels[instr_id] = ' '.join(row["tag"].split()[:2])
            instr_meths[instr_id] = row["meth"]
            instr_loc[instr_id] = lines[int(re.findall(instr_number_pattern, instr_id)[0]) - 1]

        if not INVALID_VAR_PATTERN.fullmatch(to_id):
            flow_from_inst[instr_id].append(to_id)
            var_loc[to_id] = lines[int(re.findall(instr_number_pattern, instr_id)[0]) - 1]

        if not INVALID_VAR_PATTERN.fullmatch(from_id):
            flow_to_inst[instr_id].append(from_id)
            if from_id not in var_labels:
                var_labels[from_id] = ' '.join(row["tag"].split()[2:])


def process_telemetry_df(telemetry_df):
    telemetry_df['TrainInstr']=telemetry_df['TrainCtx']+telemetry_df['TrainLine'].astype(str)+telemetry_df['TrainCtx']
    telemetry_df['TestInstr']=telemetry_df['TestCtx']+telemetry_df['TestLine'].astype(str)+telemetry_df['TestCtx']
    telemetry_df['TrainVar']=telemetry_df['TrainData']+telemetry_df['TrainCtx']
    telemetry_df['TestVar']=telemetry_df['TestData']+telemetry_df['TestCtx']
    return telemetry_df


def build_features_df(index_mapping,instr_labels,var_labels,instr_meths,var_loc,instr_loc):
    # Creating the features dataframe
    labels = pd.DataFrame(index=range(len(index_mapping)),columns=['Nodes', 'Labels', 'Code','Method'])
    for key, value in instr_labels.items():
        labels.iloc[index_mapping[key]] = {'Nodes': key, 'Labels': value, 'Code': instr_loc[key] if key in instr_loc.keys() else "", 'Method':instr_meths[key] if 'NonLocalMethod' in value else ""}
    for key, value in var_labels.items():
        labels.iloc[index_mapping[key]] = {'Nodes': key, 'Labels': value, 'Code': var_loc[key] if key in var_loc.keys() else "", 'Method':instr_meths[key] if 'NonLocalMethod' in value else ""}

    labels.fillna("",inplace=True)
    return labels

    

def create_binary_features(lpd):
    
      labels =["LoadField","LoadIndex","StoreFieldSSA","StoreIndexSSA","StoreIndex","AssignVar",
              "InterProcAssign","AssignGlobal","LoadSlice","StoreSliceSSA","LocalMeth","NonLocalMethod",
              "Base","Field","Index","Start","End","Step","Var","Param","Value","Train","_Phi_",
              "Test","Left","Right","Instr", "TrainData","TestData", "AssignBoolConstant", "AssignStrConstant",
               "AssignIntConstant","AssignFloatConstant", "Input", "Return" , "LoadExtSlice" , "StoreExtSlice",
              "Add", "Sub", "Mult", "Div", "FloorDiv","Mod","Pow","BitAnd","BitOr","BitXor","LShift","RShift",
               "Invert","Not","UAdd", "USub"
               ]
      
      mlb = MultiLabelBinarizer(classes=list(map(lambda x:x.lower(),labels)))
      features = lpd.iloc[:,1].to_list()
      features = [set(labels.lower().split()) for labels in features]
      features = mlb.fit_transform(features)
      
      return torch.from_numpy(features).to(torch.float)


def create_code_embeddings(df,tokenizer,sent2vec_model):
     features = list(map(lambda code: code.strip(),df.iloc[:,3].to_list()))
     lines_content_tokenized = [tokenizer.encode(line).tokens for line in features]
     code_embeddings = [sent2vec_model.embed_sentence(" ".join(line_content_tokenized))[0] for line_content_tokenized in lines_content_tokenized]
     return torch.from_numpy(np.array(code_embeddings)).to(torch.float)


def build_adj(flow_from_inst,flow_to_inst, unique_instr,unique_vars,index_mapping):

    adj= np.zeros((len(unique_instr)+len(unique_vars),len(unique_instr)+len(unique_vars)))
    for index,instr in enumerate(unique_instr):
        for to_vars in flow_from_inst[instr]:
            adj[index][index_mapping[to_vars]]=1
        for from_vars in flow_to_inst[instr]:
            adj[index_mapping[from_vars]][index]=1

    return adj


def get_columns(filename):
    d = {
        "FLowVarTransformation.csv": ['To', 'ToCtx', 'Instr', 'From','FromCtx', 'tag', 'meth', 'FromIdx', 'ToIdx'],
        "Telemetry_ModelPair.csv": ['TrainModel','TrainData','TrainInvo','TrainLine','TrainMethod','TrainCtx',
                                    'TestModel','TestData','TestInvo','TestLine','TestMethod','TestCtx'],
        "InvokeInjected.csv":['Invocation','Method','InMeth'],
        "FLowVarStoreIndex.csv": ['To', 'ToCtx', 'Instr', 'From','FromCtx', 'tag', 'meth', 'FromIdx', 'ToIdx'],
    }
    
    return d[filename]

def read_csv_or_empty(fact_path,filename):

    filepath= os.path.join(fact_path,filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath, sep="\t", names=get_columns(filename))
    else:
        return pd.DataFrame()



def read_index_or_empty(df,index):

    if df.empty:
        return []
    else:
        return list(df[index])
    

def match_invo(label,injected_invos):
    numbers = re.findall("\d+", label)
    strings = re.split("\d+",label)
    new_numbers = []
    for number in numbers:
        shift = sum(1 for el in injected_invos if el < int(number))
        new_numbers.append(str(int(number)-shift))
    new_label = ""
    for i,string in enumerate(strings):
      new_label += string
      if(i<len(strings)-1):
        new_label += new_numbers[i]
    return new_label


def create_and_save_dgl_subgraph(pair_node_id,G,binary_features,code_embedding,graph_paths,original):  
    sg,_ = dgl.khop_in_subgraph(G, pair_node_id,k=G.num_edges(),relabel_nodes=True)
    _ID = sg.ndata["_ID"]
    sg.ndata["features"]= torch.cat((binary_features[_ID],code_embedding[_ID]),dim = 1)
    save_graphs(os.path.join(graph_paths,f'{original}.bin'), [sg])
    return sg,_ID

def create_and_save_raw_subgraph(sg,_ID,unique_instr,unique_vars,feature_labels,graph_paths,original):
    adj_sg = dgl.khop_adj(sg,1)
    kept_nodes = [node for index, node in enumerate(unique_instr + unique_vars) if index in _ID]
    A = pd.DataFrame(adj_sg, index=kept_nodes, columns=kept_nodes)
    np.transpose(A).to_csv(os.path.join(graph_paths,f"{original}_A.csv"))
    feature_labels.iloc[_ID].to_csv(f"{graph_paths}/{original}_features.csv")


def build_subgraphs(fact_path,file_path,tokenizer,sent2vec_model):
    
    #Readind necessary files
    FlowVarTransformation= read_csv_or_empty(fact_path,"FLowVarTransformation.csv")
    FlowVarStoreIndex= read_csv_or_empty(fact_path,"FLowVarStoreIndex.csv")
    df_injected= read_csv_or_empty(fact_path,"InvokeInjected.csv")
    df_telemetry_model_pair= read_csv_or_empty(fact_path,"Telemetry_ModelPair.csv")

    #Reading the IR file
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise FileNotFoundError
    except Exception as e:
        print(f"An error occurred: {e}")
        raise Exception


    #Making sure there are models detected and dataflows
    if(df_telemetry_model_pair.empty==True or FlowVarTransformation.empty==True):
        return 
    if(FlowVarStoreIndex.empty==False):
            FlowVarStoreIndex= preprocess_flow_df(FlowVarStoreIndex)
    df_telemetry_model_pair= process_telemetry_df(df_telemetry_model_pair)
    FlowVarTransformation= preprocess_flow_df(FlowVarTransformation)
    


    #Creating the graphs folder
    if not os.path.exists(os.path.join(fact_path,'_graphs')):
        os.makedirs(os.path.join(fact_path,'_graphs'))
        print("created",fact_path+'/_graphs')
    else:
        remove_files(os.path.join(fact_path,'_graphs'))
    graph_paths= os.path.join(fact_path,"_graphs")


    # Extracting the set of unique instructions and vars 
    unique_instr,unique_vars,index_mapping = extract_instr_and_vars(FlowVarTransformation,FlowVarStoreIndex)



    # Storing data flows in a dict
    flow_from_inst={ instr:[] for instr in unique_instr}
    flow_to_inst={instr:[] for instr in unique_instr}
    instr_labels={}
    instr_loc={}
    instr_meths={}
    var_loc={}
    var_labels={}
    process_flow_df(FlowVarTransformation, lines,
                    instr_labels, instr_meths, instr_loc,
                    flow_from_inst, flow_to_inst, var_loc, var_labels)
    process_flow_df(FlowVarStoreIndex, lines,
                    instr_labels, instr_meths, instr_loc,
                    flow_from_inst, flow_to_inst, var_loc, var_labels)
    flow_from_inst = {instr:list(set(vars)) for instr,vars in flow_from_inst.items()}
    flow_to_inst = {instr:list(set(vars)) for instr,vars in flow_to_inst.items()}



    

    #Building the overall data flow graph 
    adj=build_adj(flow_from_inst,flow_to_inst, unique_instr,unique_vars,index_mapping)
    Adj_df = pd.DataFrame(adj, index=unique_instr+unique_vars, columns=unique_instr+unique_vars)
    G = dgl.graph(np.where(adj>0), num_nodes=len(adj))
    np.transpose(Adj_df).to_csv(os.path.join(graph_paths,"A_unpruned.csv"))



    #Building the labels features
    labels = build_features_df(index_mapping,instr_labels,var_labels,instr_meths,var_loc,instr_loc)
    labels.to_csv(f"{graph_paths}/features_unpruned.csv")

    


    #Building the per-TTI subgraphs
    injected_invos = list(map(lambda x : int(re.search("\d+", x).group()), df_injected["Invocation"])) if df_injected.empty == False else []
    code_embedding = create_code_embeddings(labels,tokenizer,sent2vec_model)
    for index,row in df_telemetry_model_pair.iterrows():
        original=row['TrainInvo']+"_"+row['TestInvo']+"_"+row['TrainCtx']+"_"+row['TestCtx']
        print(original)
        original = match_invo(original,injected_invos) if len(injected_invos)>0 else original
        pair = [row['TrainInstr'],row['TestInstr']]
        pair_node_id = list(map(lambda x: index_mapping[x],pair))


        feature_labels= labels.copy()
        try:
            feature_labels.iloc[index_mapping[row['TrainInstr']],1]+= ' Train'
            feature_labels.iloc[index_mapping[row['TestInstr']],1]+= ' Test'
            feature_labels.iloc[index_mapping[row['TrainVar']],1]+= ' TrainData'
            feature_labels.iloc[index_mapping[row['TestVar']],1]+= ' TestData'
        except KeyError:
            print(f'The model pair [{row["TrainModel"]},{row["TestModel"]}] out of scope, skipping it...')
            continue
        binary_features = create_binary_features(feature_labels)
            
        sg,_ID = create_and_save_dgl_subgraph(pair_node_id,G,binary_features,code_embedding,graph_paths,original)
        create_and_save_raw_subgraph(sg,_ID,unique_instr,unique_vars,feature_labels,graph_paths,original)
        


        
        
        




    




        



    

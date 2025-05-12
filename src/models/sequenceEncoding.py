from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                           tokenizer="text_sonar_basic_encoder")


def sequenceEncoder(model_name='all-MiniLM-L6-v2', device='cuda', tokens=None):
    model = SentenceTransformer(model_name, device=device)
    x = model.encode(tokens, show_progress_bar=True,
                 convert_to_tensor=True, device=device)

    return x


device='cuda'
def process_data(dataframe):
    data_list = []
    allRelList = []
    allNodeList = []
    info = ['x', 'edge_index', 'edge_attr', 'label']
    for index, row in dataframe.iterrows():

        sourceList = eval(row['source'])
        targetList = eval(row['target'])
        rels = eval(row['relation'])

        data = dict.fromkeys(info, None)

        data['label'] = row['label']

        encoded_edge_attr= sequenceEncoder(model_name='all-MiniLM-L6-v2', device=device,tokens=rels)
        #encoded_edge_attr = laser.embed_sentences(rels, lang='en')
        #encoded_edge_attr = torch.from_numpy(encoded_edge_attr).float().to(device)
        data['edge_attr']=encoded_edge_attr
      
        nodeList = sourceList + targetList
        temp_nodeDict = defaultdict(lambda: len(temp_nodeDict))
        temp_nodeId = [temp_nodeDict[ele] for ele in nodeList]

        nodes = list(temp_nodeDict.keys())
        encoded_nodes= sequenceEncoder(model_name='all-MiniLM-L6-v2', device=device,tokens=nodes)
        #encoded_nodes = laser.embed_sentences(nodes, lang='en')
        #encoded_nodes  = torch.from_numpy(encoded_nodes).float().to(device)
        data['x']=encoded_nodes

        edge_index= []
        src_ids =[]
        trg_ids =[]
        for i,source in enumerate(sourceList):
            src_id= temp_nodeDict[source]
            src_ids.append(src_id)
            trg_id= temp_nodeDict[targetList[i]]
            trg_ids.append(trg_id)

        edge_index.append(src_ids)
        edge_index.append(trg_ids)

        data['edge_index']= torch.tensor(edge_index)
        #print(data['edge_index'])

        #data_list.append(Data(x=data['x'], edge_index=data['edge_index'], edge_attr=data['edge_attr'], y=torch.tensor(row['label'])))


        temp_relDict = defaultdict(lambda: len(temp_relDict))
        temp_relId = [temp_relDict[ele2] for ele2 in rels]

        allRelList = allRelList + rels
        allNodeList = allNodeList + sourceList + targetList

    return allRelList, allNodeList

train_relList, train_nodeList = process_data(X_train) # claimReview data with AMR nodes and relations
train_loader = DataLoader(train_dataList,shuffle=False, batch_size=64)
test_dataList, test_relList, test_nodeList = process_data(X_test) # claimReview data with AMR nodes and relations
test_loader = DataLoader(test_dataList,shuffle=False, batch_size=64)

import json
import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx


json_name = ""
rnagraphs = ""
rnagraphs_y = ""



class RNAGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(RNAGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [json_name]  # JSON 文件名

    @property
    def processed_file_names(self):
        return [rnagraphs, rnagraphs_y]

    def download(self):
        pass

    def process(self):
        data_list = []
        rna_type_mapping = {
            'tRNA': 0,
            'pre_miRNA': 1,
            'SRP_RNA': 2,
            'snoRNA': 3,
            'sRNA': 4,
            'snRNA': 5,
            'rRNA': 6,
            'misc_RNA': 7,
            'ncRNA': 8,
            'Y_RNA': 9,
            'telomerase_RNA': 10,
            'vault_RNA': 11,
            'other': 12,
            'miRNA': 13,
            'RNase_P_RNA': 14,
            'lncRNA': 15,
            'hammerhead_ribozyme': 16,
            'piRNA': 17,
            'ribozyme': 18,
            'RNase_MRP_RNA': 19,
            'scaRNA': 20,
            'scRNA': 21,
            'tmRNA': 22,
            'precursor_RNA': 23
        }
        y_list = []  # 用来存储每个图的类别
        # 读取原始 JSON 数据文件
        with open(self.raw_paths[0], 'r') as f:
            rna_data = json.load(f)
            for entry in rna_data:
                sequence = entry['sequence']
                secondary_structure = entry['secondary_structure']
                id = entry['number']
                type = entry['rna_type']
                graph_data = create_rna_graph(sequence, secondary_structure, id, type)
                graph = graph_data['graph']
                node_features_list = graph_data['node_features']
                edge_index = create_edge_index(graph, node_features_list)
                edge_attr = create_edge_attr(graph)

                # 将 node_features_list 中的字典转换为张量
                node_features_tensor_list = []
                for node_features in node_features_list:
                    features = [node_features['base_type'], node_features['is_structure_node'],
                                node_features['position']]
                    node_features_tensor = torch.tensor(features, dtype=torch.float)
                    node_features_tensor_list.append(node_features_tensor)

                # 创建一个包含节点特征张量的列表
                x = torch.stack(node_features_tensor_list)

                # 创建 Data 对象，传递节点特征张量 x、边索引 edge_index 和边属性 edge_attr
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data.id = id
                data_list.append(data)

                # 存储每个图的类别
                y_list.append(rna_type_mapping[type])
            for i, data in enumerate(data_list):
                y_tensor = torch.tensor([y_list[i]], dtype=torch.long)
                print(y_tensor, 'y_tensor')
                data.y = y_tensor
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        torch.save(y_tensor, self.processed_paths[1])


# Helper function to create an RNA graph from sequence and secondary structure
def create_rna_graph(sequence, secondary_structure, rna_id, rna_type):
    base_type_mapping = {'A': 0, 'G': 1, 'U': 2, 'C': 3}
    graph = nx.Graph()
    node_features_list = []
    for i, (base, struct_char) in enumerate(zip(sequence, secondary_structure)):
        is_structure_node = (struct_char in '()')  

        node_features = {
            'base_type': base_type_mapping[base], 
            'is_structure_node': is_structure_node,  
            'position': i  
        }
        node_features_list.append(node_features)

 
        graph.add_node((rna_id, i), base_type=base_type_mapping[base], is_structure_node=is_structure_node, position=i)


        if i > 0:
            graph.add_edge((rna_id, i - 1), (rna_id, i), edge_type='sequence')


    stack = [] 
    for i, struct_char in enumerate(secondary_structure):
        if struct_char == '(':
            stack.append(i)
        elif struct_char == ')':
            if stack:
                left_parenthesis = stack.pop()
                graph.add_edge((rna_id, left_parenthesis), (rna_id, i), edge_type='structure')


    for i in range(len(sequence)):
        neighbors = list(graph.neighbors((rna_id, i)))


        structure_neighbors = [neighbor for neighbor in neighbors if
                               graph.get_edge_data((rna_id, i), neighbor)['edge_type'] == 'structure']

        neighbors = [neighbor for neighbor in neighbors if neighbor != (rna_id, i)]
        node_features_list[i]['neighbors'] = neighbors
    return {'graph': graph, 'node_features': node_features_list}


def create_edge_index(graph, node_features_list):
    edge_index = []  

    for i, node_features in enumerate(node_features_list):
        for neighbor_index in node_features['neighbors']:
            edge_index.append([int(i), int(neighbor_index[1])])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


def create_edge_attr(graph):
    edge_attr = []
    for edge in graph.edges(data=True):
        _, _, data = edge
        edge_type = data['edge_type']
        if edge_type == 'sequence':
            edge_attr.append([0]) 
        elif edge_type == 'structure':
            edge_attr.append([1])  
    return torch.tensor(edge_attr)

def choose_dataset(key):
    global json_name
    global rnagraphs
    global rnagraphs_y
    if key == "family_train":
        json_name = 'family_train.json'
        rnagraphs = 'rnagraphs_ftrain.pt'
        rnagraphs_y = 'rnagraphs_ftrain_y.pt'
    elif key == "family_dev":
        json_name = 'family_dev.json'
        rnagraphs = 'rnagraphs_fdev.pt'
        rnagraphs_y = 'rnagraphs_fdev_y.pt'
    elif key == "family_pre":
        json_name = 'family_pre.json'
        rnagraphs = 'rnagraphs_fpre.pt'
        rnagraphs_y = 'rnagraphs_fpre_y.pt'
    elif key == "pre_train":
        json_name = 'pretrain.json'
        rnagraphs = 'rnagraphs_fpre_train.pt'
        rnagraphs_y = 'rnagraphs_fpre_train_y.pt'
    dataset = RNAGraphDataset(root='/root/autodl-tmp/no-codingRNA-pretrain/main/graph_fea/ncRNAdata')
    return dataset


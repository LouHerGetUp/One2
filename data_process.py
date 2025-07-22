import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader
from sklearn.model_selection import train_test_split
import gc


def adj(index, ip_type, df):
    list_flow = []
    list_ip = []
    for i in range(df.shape[0]):
        list_flow.append(df.loc[i, index])
        list_ip.append(df.loc[i, ip_type])
    adj_mat = np.array([list_flow, list_ip])
    return adj_mat


def data_process(path, dataset, binary, batchsize):
    print("data_process begin!")

    chosen_features = ['Source Port', 'Destination Port', 'Bwd Packet Length Min', 'Subflow Fwd Packets',
                       'Total Length of Fwd Packets', 'Fwd Packet Length Mean',
                       'Total Length of Fwd Packets',
                       'Fwd Packet Length Std', 'Fwd IAT Min', 'Flow IAT Min', 'Flow IAT Mean',
                       'Bwd Packet Length Std',
                       'Subflow Fwd Bytes', 'Flow Duration', 'Flow IAT Std', 'Active Min', 'Active Mean',
                       'Bwd IAT Mean',
                       'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'ACK Flag Count', 'Fwd PSH Flags',
                       'SYN Flag Count',
                       'Flow Packets/s', 'PSH Flag Count', 'Average Packet Size']

    print("read_csv begin!")
    df = pd.read_csv(path)
    print("read_csv done!")

    df_y = df['Label']

    if dataset in {'CIC-IDS2017', 'CIC-UNSW-NB15'}:
        df = df.join(pd.get_dummies(df['Protocol'], dtype='float'))
        df_flow = df.drop(
            columns=['Source IP', 'Destination IP', 'Label', 'index', 'Source Port', 'Destination Port', 'Protocol'])

        Source_Port_list = df.groupby(['Source IP', 'Source Port']).count().index.to_list()
        indices = range(len(Source_Port_list))
        zip_iterator = zip(Source_Port_list, indices)
        Source_Port_dict = dict(zip_iterator)

        Destination_Port_list = df.groupby(['Destination IP', 'Destination Port']).count().index.to_list()
        indices = range(len(Destination_Port_list))
        zip_iterator = zip(Destination_Port_list, indices)
        Destination_Port_dict = dict(zip_iterator)

        df['Source IP and Port'] = df.apply(lambda x: tuple([x['Source IP'], x['Source Port']]), axis=1)
        df['Source Port Unique'] = df['Source IP and Port'].map(lambda x: Source_Port_dict.get(x))
        df['Destination IP and Port'] = df.apply(lambda x: tuple([x['Destination IP'], x['Destination Port']]), axis=1)
        df['Destination Port Unique'] = df['Destination IP and Port'].map(lambda x: Destination_Port_dict.get(x))

    elif dataset in {'CAR-HACKING', 'CAN-intrusion', 'CICIoV2024'}:
        df_flow = df.drop(columns=['DLC', 'Label', 'ID'])

    elif dataset == 'TON_IoT':
        df = df.join(pd.get_dummies(df['Protocol'], prefix='Protocol', dtype='float'))
        df_flow = df.drop(
            columns=['Source IP', 'Destination IP', 'Label', 'index', 'Source Port', 'Destination Port', 'Protocol'])
        df_flow = df_flow.drop(columns=['type', 'label'])

        print("groupby begin!")
        Source_Port_list = df.groupby(['Source IP', 'Source Port']).count().index.to_list()
        indices = range(len(Source_Port_list))
        zip_iterator = zip(Source_Port_list, indices)
        Source_Port_dict = dict(zip_iterator)

        Destination_Port_list = df.groupby(['Destination IP', 'Destination Port']).count().index.to_list()
        indices = range(len(Destination_Port_list))
        zip_iterator = zip(Destination_Port_list, indices)
        Destination_Port_dict = dict(zip_iterator)

        df['Source IP and Port'] = df.apply(lambda x: tuple([x['Source IP'], x['Source Port']]), axis=1)
        df['Source Port Unique'] = df['Source IP and Port'].map(lambda x: Source_Port_dict.get(x))
        df['Destination IP and Port'] = df.apply(lambda x: tuple([x['Destination IP'], x['Destination Port']]), axis=1)
        df['Destination Port Unique'] = df['Destination IP and Port'].map(lambda x: Destination_Port_dict.get(x))
        print("groupby done!")

        print("get_dummies begin!")
        for col in df_flow.columns:
            if df_flow[col].dtypes == 'object':
                if len(df_flow[col].value_counts()) <= 3:
                    df_flow = df_flow.join(pd.get_dummies(df_flow[col], prefix=col, dtype='float'))
                df_flow = df_flow.drop(columns=[col])
        print("get_dummies done!")

    else:
        pass

    for i in list(df_flow.columns):
        Max = np.max(df_flow[i])
        Min = np.min(df_flow[i])
        if Max == Min:
            df_flow[i] = 0
        else:
            df_flow[i] = (df_flow[i] - Min) / (Max - Min)

    print("adj begin!")
    if dataset in {'CIC-IDS2017', 'TON_IoT', 'CIC-UNSW-NB15'}:
        flow_to_Source_IP_adj = adj('index', 'Source IP', df)
        flow_to_Destination_IP_adj = adj('index', 'Destination IP', df)
        Source_IP_to_flow_adj = adj('Source IP', 'index', df)
        Destination_IP_to_flow_adj = adj('Destination IP', 'index', df)

        flow_to_Source_Port_adj = adj('index', 'Source Port Unique', df)
        flow_to_Destination_Port_adj = adj('index', 'Destination Port Unique', df)
        Source_Port_to_flow_adj = adj('Source Port Unique', 'index', df)
        Destination_Port_to_flow_adj = adj('Destination Port Unique', 'index', df)

    elif dataset in {'CAR-HACKING', 'CAN-intrusion', 'CICIoV2024'}:
        flow_to_ID = adj('index', 'ID', df)
        ID_to_flow = adj('ID', 'index', df)

        print(flow_to_ID)
        print(type(flow_to_ID))
        print(flow_to_ID.shape)

    print("adj done!")

    train_mask = np.zeros([df.shape[0]], dtype=int)
    val_mask = np.zeros([df.shape[0]], dtype=int)
    test_mask = np.zeros([df.shape[0]], dtype=int)

    for i in range(int(train_mask.shape[0] * 0.6)):
        train_mask[i] = 1
    for i in range(int(val_mask.shape[0] * 0.6), int(val_mask.shape[0] * 0.8)):
        val_mask[i] = 1
    for i in range(int(test_mask.shape[0] * 0.8), int(test_mask.shape[0])):
        test_mask[i] = 1

    X_train, X_test, y_train, y_test = train_test_split(df_flow, df_y, test_size=0.2, random_state=1999, stratify=df_y)
    train_mask_list = y_train.index.to_list()
    test_mask_list = y_test.index.to_list()
    for i in train_mask_list:
        train_mask[i] = 1
    for i in test_mask_list:
        test_mask[i] = 1

    if binary == True:
        df['Label'] = [0 if k == 0 else 1 for k in df['Label']]
        df_y = df['Label']

    print("dataset begin!")
    data = HeteroData()

    if dataset in {'CIC-IDS2017', 'TON_IoT', 'CIC-UNSW-NB15'}:
        num_src_ip = len(df['Source IP'].value_counts().index.to_list())
        num_dst_ip = len(df['Destination IP'].value_counts().index.to_list())
        num_src_port = len(df['Source Port Unique'].value_counts().index.to_list())
        num_dst_port = len(df['Destination Port Unique'].value_counts().index.to_list())

        print(len(df))
        print(min(df['index']))
        print(max(df['index']))
        print(min(df['Source IP'].value_counts().index.values))
        print(max(df['Source IP'].value_counts().index.values))
        print(df['Source IP'].value_counts().index.values.shape)
        print(min(df['Destination IP'].value_counts().index.values))
        print(max(df['Destination IP'].value_counts().index.values))
        print(df['Destination IP'].value_counts().index.values.shape)
        print(min(df['Source Port Unique'].value_counts().index.values))
        print(max(df['Source Port Unique'].value_counts().index.values))
        print(df['Source Port Unique'].value_counts().index.values.shape)
        print(min(df['Destination Port Unique'].value_counts().index.values))
        print(max(df['Destination Port Unique'].value_counts().index.values))
        print(df['Destination Port Unique'].value_counts().index.values.shape)

    elif dataset in {'CAR-HACKING', 'CAN-intrusion', 'CICIoV2024'}:
        num_id = len(df['ID'].value_counts().index.to_list())

    num_flow_features = df_flow.values.shape[1]

    data['flow'].x = torch.tensor(df_flow.values, dtype=torch.float32)
    data['flow'].y = torch.tensor(df_y.values, dtype=torch.int64)
    data['flow'].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data['flow'].test_mask = torch.tensor(test_mask, dtype=torch.bool)

    if dataset in {'CIC-IDS2017', 'TON_IoT', 'CIC-UNSW-NB15'}:
        data['src_ip'].x = torch.tensor(np.eye(num_src_ip), dtype=torch.float32)
        data['dst_ip'].x = torch.tensor(np.eye(num_dst_ip), dtype=torch.float32)

        data['src_port'].x = torch.tensor(np.eye(num_src_port), dtype=torch.float32)
        data['dst_port'].x = torch.tensor(np.eye(num_dst_port), dtype=torch.float32)

        data['flow', 'src_ip'].edge_index = torch.tensor(flow_to_Source_IP_adj, dtype=torch.int64)
        data['flow', 'dst_ip'].edge_index = torch.tensor(flow_to_Destination_IP_adj, dtype=torch.int64)
        data['src_ip', 'flow'].edge_index = torch.tensor(Source_IP_to_flow_adj, dtype=torch.int64)
        data['dst_ip', 'flow'].edge_index = torch.tensor(Destination_IP_to_flow_adj, dtype=torch.int64)

        data['flow', 'src_port'].edge_index = torch.tensor(flow_to_Source_Port_adj, dtype=torch.int64)
        data['flow', 'dst_port'].edge_index = torch.tensor(flow_to_Destination_Port_adj, dtype=torch.int64)
        data['src_port', 'flow'].edge_index = torch.tensor(Source_Port_to_flow_adj, dtype=torch.int64)
        data['dst_port', 'flow'].edge_index = torch.tensor(Destination_Port_to_flow_adj, dtype=torch.int64)

        metapaths = [[('flow', 'src_ip'), ('src_ip', 'flow')],
                     [('flow', 'dst_ip'), ('dst_ip', 'flow')]]

    elif dataset in {'CAR-HACKING', 'CAN-intrusion', 'CICIoV2024'}:
        print(df['ID'].value_counts().index.values)
        print(df['ID'].value_counts().index.values.shape)
        print(np.count_nonzero(df['ID'].value_counts().index.values > 40))
        print(type(df['ID'].value_counts().index.values))

        data['id'].x = torch.tensor(np.expand_dims(df['ID'].value_counts().index.values, 1),
                                    dtype=torch.float32)

        data['flow', 'id'].edge_index = torch.tensor(flow_to_ID, dtype=torch.int64)
        data['id', 'flow'].edge_index = torch.tensor(ID_to_flow, dtype=torch.int64)

        print("data['flow'].num_nodes={}".format(data['flow'].num_nodes))
        print("data['flow', 'id'].num_edges={}".format(data['flow', 'id'].num_edges))
        print(data)

        metapaths = [[('flow', 'id'), ('id', 'flow')]]

    print("dataset done!")

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    data_loader = HGTLoader(data, num_samples={key: [1024] * 4 for key in data.node_types},
                            batch_size=batchsize, input_nodes=('flow', data['flow'].train_mask),
                            shuffle=True, num_workers=16)

    print("data_loader done!")

    return data_loader
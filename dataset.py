import numpy as np
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------
# Dataset
# -----------------------------------------
class CustomDataset_xsy(Dataset):
    '''for tabular train/test'''
    def __init__(self, file_name, data_name, sensitive):
        if data_name == 'adult':
            self.target = "income_ >50K"
        elif data_name == 'german':
            self.target = "risk_Bad"
        else:
            raise ValueError

        self.sensitive = sensitive
        self.df = pd.read_csv(file_name)
        self.y = pd.DataFrame(self.df[self.target])
        self.s = pd.DataFrame(self.df[self.sensitive])
        self.X = self.df.drop([self.target, self.sensitive], axis=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # (N, D), (N, 1), (N, 1)
        return torch.FloatTensor(np.array(self.X.iloc[idx])), \
               torch.FloatTensor(np.array(self.s.iloc[idx])), \
               torch.FloatTensor(np.array(self.y.iloc[idx]))


class NoisySet(Dataset):
    '''make noisy tabular'''
    def __init__(self, file_name, data_name, sensitive, args):
        if data_name == 'adult':
            self.target = "income_ >50K"
        elif data_name == 'german':
            self.target = "risk_Bad"
        else:
            raise ValueError

        self.sensitive = sensitive
        self.df = pd.read_csv(file_name)
        self.data_flag = file_name.split('/')[-1].split('_')[1]
        self.make_env(args)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        # (N, D), (N, 1), (N, 1)
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.s[idx]), torch.FloatTensor(self.y[idx])

    def make_env(self, args):
        # --------------------------------------- 1. reduce trainset
        if self.data_flag == 'train':
            if args.tr_ratio < 0.999:
                X_train, _, y_train, _ = train_test_split(self.df.iloc[:, :-1], self.df.iloc[:, -1], shuffle=True,
                                                            random_state=args.seed, train_size=args.tr_ratio,
                                                            stratify=self.df[[args.sensitive, args.target]])                        
                self.y = np.array(y_train).reshape(-1, 1)
                self.s = np.array(X_train[self.sensitive]).reshape(-1, 1)
                self.X = np.array(X_train.drop([self.sensitive], axis=1))
                del X_train; del y_train
            else:
                # use full train set
                self.y = np.array(self.df[self.target]).reshape(-1, 1)
                self.s = np.array(self.df[self.sensitive]).reshape(-1, 1)
                self.X = np.array(self.df.drop([self.target, self.sensitive], axis=1))
        else:
            self.y = np.array(self.df[self.target]).reshape(-1, 1)
            self.s = np.array(self.df[self.sensitive]).reshape(-1, 1)
            self.X = np.array(self.df.drop([self.target, self.sensitive], axis=1))

        # --------------------------------------- 2. add label noise
        if args.env_flag == 'nn':
            pass
        else:
            if self.data_flag == 'train':
                s_n = np.abs(self.s - np.random.binomial(1, args.env_eps_s_tr, size=self.s.shape))
                y_n = np.abs(self.y - np.random.binomial(1, args.env_eps_y_tr, size=self.y.shape))
                self.s = self.s if args.env_flag[0] == 'n' else s_n
                self.y = self.y if args.env_flag[1] == 'n' else y_n
                del s_n
                del y_n


class RepresentationSet(Dataset):
    '''for 2 stage evaluation'''
    def __init__(self, z, file_name, target, sensitive):
        Data = pd.read_csv(file_name)
        self.z = z
        self.s = pd.DataFrame(Data[sensitive])
        self.y = pd.DataFrame(Data[target])


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return torch.FloatTensor(self.z[idx]), \
               torch.FloatTensor(np.array(self.s.iloc[idx])), \
               torch.FloatTensor(np.array(self.y.iloc[idx]))


class NoisyReprSet(Dataset):
    def __init__(self, z, dataset):
        self.z = z
        self.s = dataset.s
        self.y = dataset.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return torch.FloatTensor(self.z[idx]), torch.FloatTensor(self.s[idx]), torch.FloatTensor(self.y[idx])


# -----------------------------------------
# DataLoader
# -----------------------------------------
def get_representation_loader(z_train, z_test, trainset, testset, test_batch_size):
    dataset_train = NoisyReprSet(z_train, trainset)
    train_batch_size = dataset_train.y.shape[0]
    dataloader_train = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, drop_last=False, num_workers=0)
    dataset_test = NoisyReprSet(z_test, testset)
    dataloader_test = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=0)
    return dataloader_train, dataloader_test


def get_xsy_loaders(train_file_name, test_file_name, data_name, sensitive, test_batch_size, args):
    trainset, testset = NoisySet(train_file_name, data_name, sensitive, args), NoisySet(test_file_name, data_name, sensitive, args)
    train_batch_size = trainset.y.shape[0]
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, drop_last=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=0)
    return trainloader, testloader

# -----------------------------------------
# YaleB Dataset
# -----------------------------------------
class PrivacyDataLoader:
    def __init__(self, train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label,
                 trainset, testset):
        self.train_data = train_data
        self.train_label = train_label
        self.train_sensitive_label = train_sensitive_label
        self.test_data = test_data, test_label
        self.test_label = test_label
        self.test_sensitive_label = test_sensitive_label
        self.trainset = trainset
        self.testset = testset


class YaleBDatasetPair(Dataset):
    def __init__(self, x, y, s, device):
        Dataset.__init__(self)
        assert x.size(0) == y.size(0)
        self.x = x.to(device)
        self.y = y.to(device)
        self.s = s.to(device)

    def __getitem__(self, index):
        x_ori = self.x[index]
        y_ori = self.y[index]
        s_ori = self.s[index]

        r = torch.randint(1, 5, size=y_ori.shape)
        cont_index = (s_ori + r) % 5 + 5 * y_ori

        x_cont = self.x[cont_index]
        s_cont = self.s[cont_index]
        y_cont = self.y[cont_index]

        # one hot enc s label
        s_ori = F.one_hot(s_ori, num_classes=5)
        s_cont = F.one_hot(s_cont, num_classes=5)

        # (N, 504), (N, 504), (N, 5), (N, 5), (N, )
        return x_ori, x_cont, s_ori, s_cont, y_ori

    def __len__(self):
        return self.x.size(0)


class YaleBDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, sensitive_tensor, device):
        Dataset.__init__(self)
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor.to(device)
        self.target_tensor = target_tensor.to(device)
        self.sensitive_tensor = sensitive_tensor.to(device)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        X = self.data_tensor[index]
        s = self.sensitive_tensor[index]
        y = self.target_tensor[index]

        s = F.one_hot(s, num_classes=5)
        # (N, 504), (N, 5), (N, )
        return X, s, y

    def __len__(self):
        return self.data_tensor.size(0)


class ExtendedYaleBDataLoader(PrivacyDataLoader):
    def __init__(self, data_path, train_data=None, train_label=None, train_sensitive_label=None, test_data=None, test_label=None,
                 test_sensitive_label=None, trainset=None, testset=None, args=None, device=None):
        self.name = "yale"
        self.data_path = data_path
        self.args=args
        self.device = device
        super().__init__(train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label,
                         trainset, testset)

    def load(self):
        data1 = pickle.load(open(os.path.join(self.data_path, "set_0.pdata"), "rb"), encoding='latin1')
        data2 = pickle.load(open(os.path.join(self.data_path, "set_1.pdata"), "rb"), encoding='latin1')
        data3 = pickle.load(open(os.path.join(self.data_path, "set_2.pdata"), "rb"), encoding='latin1')
        data4 = pickle.load(open(os.path.join(self.data_path, "set_3.pdata"), "rb"), encoding='latin1')
        data5 = pickle.load(open(os.path.join(self.data_path, "set_4.pdata"), "rb"), encoding='latin1')
        test = pickle.load(open(os.path.join(self.data_path, "test.pdata"), "rb"), encoding='latin1')
        train_data = np.concatenate((data1["x"], data2["x"], data3["x"], data4["x"], data5["x"]), axis=0)
        train_label = np.concatenate((data1["t"], data2["t"], data3["t"], data4["t"], data5["t"]), axis=0)
        train_sensitive_label = np.concatenate(
            (data1["light"], data2["light"], data3["light"], data4["light"], data5["light"]), axis=0)
        test_data = test["x"]
        test_label = test["t"]
        test_sensitive_label = test["light"]

        index = test_sensitive_label != 5
        test_label = test_label[index]
        test_sensitive_label = test_sensitive_label[index]
        test_data = test_data[index]

        self.n_target_class = 38
        self.n_sensitive_class = 5

        self.train_size = train_data.shape[0]
        self.test_size = test_data.shape[0]

        # trainset setup. sort by person ID
        self.train_data = torch.from_numpy(train_data).float()
        self.train_label = torch.from_numpy(train_label).long()
        self.train_sensitive_label = torch.from_numpy(train_sensitive_label).long()

        self.sort_index = torch.argsort(self.train_label)
        self.train_data = self.train_data[self.sort_index] / 255
        self.train_label = self.train_label[self.sort_index]
        self.train_sensitive_label = self.train_sensitive_label[self.sort_index]

        # sort by light condition again
        for i in range(0, 190, 5):
            sort_index = torch.argsort(self.train_sensitive_label[i:i + 5])
            sort_index = sort_index + i
            self.train_data[i:i + 5] = self.train_data[sort_index]
            self.train_label[i:i + 5] = self.train_label[sort_index]
            self.train_sensitive_label[i:i + 5] = self.train_sensitive_label[sort_index]

        # testset setup
        self.test_data = torch.from_numpy(test_data).float()
        self.test_label = torch.from_numpy(test_label).long()
        self.test_sensitive_label = torch.from_numpy(test_sensitive_label).long()
        self.test_data = self.test_data / 255

        # add noise according to env flag
        self.make_env(self.args)

        self.pretrainset = YaleBDatasetPair(self.train_data, self.train_label, self.train_sensitive_label, self.device)
        self.trainset = YaleBDataset(self.train_data, self.train_label, self.train_sensitive_label, self.device)
        self.testset = YaleBDataset(self.test_data, self.test_label, self.test_sensitive_label, self.device)

    def make_env(self, args):
        if args.env_flag == 'nn':
            pass
        else: 
            # -------------------------------------- 1. reduce train set
            if args.tr_ratio < 0.999:
                self.train_label = self.train_label.reshape(-1, 1)
                self.train_sensitive_label = self.train_sensitive_label.reshape(-1, 1)

                rng_state = np.random.get_state()
                np.random.shuffle(self.train_data)
                np.random.set_state(rng_state)              
                np.random.shuffle(self.train_label)
                np.random.set_state(rng_state)              
                np.random.shuffle(self.train_sensitive_label)

                self.train_data = self.train_data[:int(self.train_label.shape[0] * args.tr_ratio), :]
                self.train_label = self.train_label[:int(self.train_label.shape[0] * args.tr_ratio), :]
                self.train_sensitive_label = self.train_sensitive_label[:int(self.train_label.shape[0] * args.tr_ratio), :]

                self.train_label = self.train_label.reshape(-1)
                self.train_sensitive_label = self.train_sensitive_label.reshape(-1)
            
            # -------------------------------------- 2. add label noise
            s_tr_p = torch.ones_like(self.train_sensitive_label) * args.env_eps_s_tr
            y_tr_p = torch.ones_like(self.train_label) * args.env_eps_y_tr
            s_tr_mask, y_tr_mask = torch.bernoulli(s_tr_p), torch.bernoulli(y_tr_p)

            s_tr_n = (self.train_sensitive_label + s_tr_mask * torch.randint(1, 5, size=s_tr_p.shape)) % 5
            y_tr_n = (self.train_label + y_tr_mask * torch.randint(1, 38, size=y_tr_p.shape)) % 38
            
            self.train_sensitive_label = self.train_sensitive_label if args.env_flag[0] == 'n' else s_tr_n.long()
            self.train_label = self.train_label if args.env_flag[1] == 'n' else y_tr_n.long()

            del s_tr_p; del s_tr_mask; del s_tr_n
            del y_tr_p; del y_tr_mask; del y_tr_n
from itertools import permutations
import numpy as np
import os.path as osp
import torch
from torch.utils.data.dataset import Dataset
import torchvision


class FullMnist(Dataset):
    def __init__(self, which='train', zero_at=0.15, one_at=2., invert=True):
        self.cs = []
        self.vals = []
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if which == 'train':
            self.data = torchvision.datasets.MNIST('../data/mnist', train=True, download=True)
            for i in range(0, 50000, 1):
                self.cs.append(self.data.train_labels[i])
                dat_flat = self.data.train_data[i].flatten().double()
                dat_flat *= 1./256.
                if invert:
                    dat_flat = dat_flat*(-1.) + 1.  # make invert white and black
                dat_flat = zero_at + dat_flat * (one_at - zero_at)
                self.vals.append(dat_flat)
        elif which == 'val':
            self.data = torchvision.datasets.MNIST('../data/mnist', train=True, download=True)
            for i in range(50000, 60000, 1):
                self.cs.append(self.data.train_labels[i])
                dat_flat = self.data.train_data[i].flatten().double()
                dat_flat *= 1./256.
                if invert:
                    dat_flat = dat_flat*(-1.) + 1.  # make invert white and black
                dat_flat = zero_at + dat_flat * (one_at - zero_at)
                self.vals.append(dat_flat)
        elif which == 'test':
            self.data = torchvision.datasets.MNIST('../data/mnist', train=False, download=True)
            for i in range(10000):
                self.cs.append(self.data.train_labels[i])
                dat_flat = self.data.train_data[i].flatten().double()
                dat_flat *= 1./256.
                if invert:
                    dat_flat = dat_flat*(-1.) + 1.  # make invert white and black
                dat_flat = zero_at + dat_flat * (one_at - zero_at)
                self.vals.append(dat_flat)

    def __getitem__(self, index):
        return self.vals[index], self.cs[index]

    def __len__(self):
        return len(self.cs)


class HicannMnist(Dataset):
    def __init__(self, which='train', width_pixel=16, zero_at=0.15, one_at=2., invert=True, late_at_inf=False):
        self.cs = []
        self.vals = []
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        if not np.all([osp.isfile(f"../data/{width_pixel}x{width_pixel}_mnist_{which}{i}.npy")
                       for i in ['_label', '']]):
            if which == 'train':
                train = True
                start_sample = 0
                end_sample = 50000
            elif which == 'val':
                train = True
                start_sample = 50000
                end_sample = 60000
            elif which == 'test':
                train = False
                start_sample = 0
                end_sample = 10000
            self.data = torchvision.datasets.MNIST(
                '../data/mnist', train=train, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop((24, 24)),
                    torchvision.transforms.Resize((width_pixel, width_pixel)),
                    torchvision.transforms.ToTensor()]))

            loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False)
            tmp_for_save = []
            for i, elem in enumerate(loader):
                if i >= start_sample and i < end_sample:
                    label = elem[1][0].data.item()
                    self.cs.append(label)
                    dat_flat = elem[0][0][0].flatten().double()
                    tmp_for_save.append(dat_flat)
                    if invert:
                        dat_flat = dat_flat * (-1.) + 1.  # make invert white and black
                    dat_flat = zero_at + dat_flat * (one_at - zero_at)
                    if late_at_inf:
                        dat_flat[dat_flat == one_at] = np.inf
                    self.vals.append(dat_flat)
            tmp_for_save = np.array([ii.cpu().detach().numpy() for ii in tmp_for_save])
            np.save(f"../data/{width_pixel}x{width_pixel}_mnist_{which}_label.npy",
                    torch.tensor(self.cs).cpu().detach().numpy())
            np.save(f"../data/{width_pixel}x{width_pixel}_mnist_{which}.npy",
                    torch.tensor(tmp_for_save).cpu().detach().numpy())
            print("Saved processed images")
        else:
            print("load preprocessed data")
            self.cs = np.load(f"../data/{width_pixel}x{width_pixel}_mnist_{which}_label.npy")
            tmp_data = np.load(f"../data/{width_pixel}x{width_pixel}_mnist_{which}.npy")
            for i, dat_flat in enumerate(tmp_data):
                if invert:
                    dat_flat = dat_flat * (-1.) + 1.  # make invert white and black
                dat_flat = zero_at + dat_flat * (one_at - zero_at)
                if late_at_inf:
                    dat_flat[dat_flat == one_at] = np.inf
                self.vals.append(dat_flat)

    def __getitem__(self, index):
        return self.vals[index], self.cs[index]

    def __len__(self):
        return len(self.cs)


class YinYangDataset(Dataset):
    def __init__(self, which='train', early=0.15, late=2.,
                 r_small=0.1, r_big=0.5, size=1000, seed=42,
                 multiply_input_layer=1):
        assert type(multiply_input_layer) == int
        self.cs = []
        self.vals = []

        try:
            import yin_yang_data_set.dataset
        except (ModuleNotFoundError, ImportError):
            print("Make sure you installed the submodule (github.com/lkriener/yin_yang_data_set)")
            raise
        tmp_dataset = yin_yang_data_set.dataset.YinYangDataset(r_small, r_big, size, seed)
        self.class_names = tmp_dataset.class_names
        loader = torch.utils.data.DataLoader(tmp_dataset, batch_size=1, shuffle=False)

        tmp_for_save = []
        for i, elem in enumerate(loader):
            self.cs.append(elem[1][0].data.item())

            # extract and multiply (used on hardware)
            vals = elem[0][0].flatten().double().repeat(multiply_input_layer)
            # transfrom values in 0..1 to spiketimes in early..late
            self.vals.append(early + vals * (late - early))

    def __getitem__(self, index):
        return self.vals[index], self.cs[index]

    def __len__(self):
        return len(self.cs)


class XOR(Dataset):
    def __init__(self, which='train', early=0.15, late=2.,
                 r_small=0.1, r_big=0.5, size=1000, seed=42,
                 multiply_input_layer=1):
        assert type(multiply_input_layer) == int
        self.cs = []
        self.vals = []
        self.class_names = ['False', 'True']

        for i, elem in enumerate(
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1],
             ]
        ):
            self.cs.append(torch.tensor(elem).sum() % 2)
            self.vals.append(
                torch.hstack([torch.tensor(elem),
                              torch.tensor([0, 1])]  # bias
                            ) * (late - early) + early
            )

    def __getitem__(self, index):
        return self.vals[index], self.cs[index]

    def __len__(self):
        return len(self.cs)

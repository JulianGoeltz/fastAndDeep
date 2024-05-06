from itertools import permutations
import numpy as np
import os.path as osp
import torch
from torch.utils.data.dataset import Dataset
import torchvision


class BarsDataset(Dataset):
    class_names = ['horiz', 'vert', 'diag']

    def __init__(self, square_size,
                 early=0.05, late=0.5,
                 noise_level=1e-2,
                 samples_per_class=10,
                 multiply_input_layer=1):
        assert type(multiply_input_layer) == int
        debug = False
        self.__vals = []
        self.__cs = []
        ones = list(np.ones(square_size) + (late - 1.))
        if debug:
            print(ones)
        starter = [ones]
        for _ in range(square_size - 1):
            starter.append(list(np.zeros(square_size) + early))
        if debug:
            print('Starter')
            print(starter)
        horizontals = []
        for h in permutations(starter):
            horizontals.append(list(h))
        horizontals = np.unique(np.array(horizontals), axis=0)
        if debug:
            print('Horizontals')
            print(horizontals)
        verticals = []
        for h in horizontals:
            v = np.transpose(h)
            verticals.append(v)
        verticals = np.array(verticals)
        if debug:
            print('Verticals')
            print(verticals)
        diag = [late - early for _ in range(square_size)]
        first = np.diag(diag) + early
        second = first[::-1]
        diagonals = [first, second]
        if debug:
            print('Diagonals')
            print(diagonals)
        n = 0
        idx = 0
        while n < samples_per_class:
            h = horizontals[idx].flatten()
            h = list(h + np.random.rand(len(h)) * noise_level)
            self.__vals.append(h)
            self.__cs.append(0)
            n += 1
            idx += 1
            if idx >= len(horizontals):
                idx = 0
        n = 0
        idx = 0
        while n < samples_per_class:
            v = verticals[idx].flatten()
            v = list(v + np.random.rand(len(v)) * noise_level)
            self.__vals.append(v)
            self.__cs.append(1)
            n += 1
            idx += 1
            if idx >= len(verticals):
                idx = 0
        n = 0
        idx = 0
        while n < samples_per_class:
            d = diagonals[idx].flatten()
            d = list(d + np.random.rand(len(d)) * noise_level)
            self.__vals.append(d)
            self.__cs.append(2)
            n += 1
            idx += 1
            if idx >= len(diagonals):
                idx = 0

        if multiply_input_layer > 1:
            self.__vals = np.array(self.__vals).repeat(multiply_input_layer, 1)

    def __getitem__(self, index):
        return np.array(self.__vals[index]), self.__cs[index]

    def __len__(self):
        return len(self.__cs)


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


class MorseDataset():
    def __init__(self, size=1000, seed=42,isi = 1.,inter_sample_noise = (0.,0.), isi_noise = (0.,0.) ):
        # if the file already exist, load else create and save in a file.
        # Define Morse code mappings for all 26 characters
        '''
        which: str - 'train', 'val' or 'test'
        size: int - number of samples per class, default 1000. 
        seed: int - random seed for reproducibility, default 42.
        isi: int - inter-spike interval, default 1.
        inter_sample_noise: tuple - mean and standard deviation of the noise added to the intial time, default (1,0.1).
        isi_noise: tuple - mean and standard deviation of the noise added to the inter-spike interval, default (1,0.1).
        if isi noise is zero and inter_sample_noise is non-zero, each sample with be shifted by the same amount of time but the consecutive spike will be at the periodic.

        '''
        morse_code_map = {
            'A': '.-',   'B': '-...', 'C': '-.-.', 'D': '-..',   'E': '.',
            'F': '..-.', 'G': '--.',  'H': '....', 'I': '..',    'J': '.---',
            'K': '-.-',  'L': '.-..', 'M': '--',   'N': '-.',    'O': '---',
            'P': '.--.', 'Q': '--.-', 'R': '.-.',  'S': '...',   'T': '-',
            'U': '..-',  'V': '...-', 'W': '.--',  'X': '-..-',  'Y': '-.--',
            'Z': '--..'
        }
        self.__vals = []
        self.__cs = []
        max_length = max(len(code) for code in morse_code_map.values())
        np.random.seed(seed)
        for _ in range(size):
            for char, code in morse_code_map.items():
                initial_time = 0. + np.random.normal(*inter_sample_noise)
                jitter = np.random.normal(*isi_noise)      
                isi_sample = isi + np.around(jitter,2)
                time = initial_time
                spikes = [[],[]]
                for symbol in code:
                    if symbol == '.': 
                        spikes[0].append(time)
                        spikes[1].append(float('inf'))                    
                    else:
                        spikes[0].append(float('inf'))
                        spikes[1].append(time)
                    
                    time += isi_sample
                spikes[0] += [float('inf')] * (max_length - len(spikes[0]))
                spikes[1] += [float('inf')] * (max_length - len(spikes[1]))
                self.__vals.append(spikes)
                self.__cs.append(char)
        
        
    def __getitem__(self, index):
        return self.vals[index], self.cs[index]

    def __len__(self):
        return len(self.cs)

class XOR(Dataset):
    def __init__(self, which='train', early=0.15, late=2.,
                 r_small=0.1, r_big=0.5, size=1000, seed=42,
                 multiply_input_layer=1):
        assert multiply_input_layer == 1
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

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


class STL_data():
    def __init__(self, args):
        super(STL_data).__init__()
        self.args = args
        self.filepath = './dataset/STL/'
        self.download = True
        self.valid_ratio = args.valid_ratio
        
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.n_workers = args.n_workers
    
    def load_data(self):
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.train_valid_dataset = datasets.STL10(root=self.filepath, split='train', download=self.download, transform=self.transform)
        self.test_dataset = datasets.STL10(root=self.filepath, split='test', download=self.download, transform=self.transform)

        train_indices, valid_indices, _, _ = train_test_split(range(len(self.train_valid_dataset)),
                                                            self.train_valid_dataset.labels,
                                                            stratify = self.train_valid_dataset.labels,
                                                            test_size = self.valid_ratio)

        self.train_dataset = Subset(self.train_valid_dataset, train_indices)
        self.valid_dataset = Subset(self.train_valid_dataset, valid_indices)

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.n_workers)
        self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.n_workers)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.n_workers)
        
        return self.train_loader, self.valid_loader, self.test_loader
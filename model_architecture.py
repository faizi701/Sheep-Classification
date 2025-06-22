from torch.utils.data import Dataset
from PIL import Image

class SheepDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.loc[idx, 'filepath']).convert('RGB')
        label = self.df.loc[idx, 'label_encoded']
        if self.transform:
            image = self.transform(image)
        return image, label

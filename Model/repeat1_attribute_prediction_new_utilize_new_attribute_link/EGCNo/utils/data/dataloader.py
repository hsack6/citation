from torch.utils.data import DataLoader
class BADataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(BADataloader, self).__init__(*args, **kwargs)
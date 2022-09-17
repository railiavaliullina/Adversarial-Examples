from torch.utils.data import DataLoader
from data.dataset import SOPDataset


def get_dataloader(cfg, dataset_type):
    """
    Get dataloader within dataset
    :param cfg: cfg['data'] part of config
    :param dataset: dataset to get dataloader from
    :return: dataLoader
    """
    dataset = SOPDataset(cfg, dataset_type)
    dl = DataLoader(dataset.image_folder,
                    batch_size=cfg['dataloader']['batch_size'][dataset_type],
                    shuffle=False,
                    drop_last=cfg['dataloader']['drop_last'][dataset_type])
    return dl

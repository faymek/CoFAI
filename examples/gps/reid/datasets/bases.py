from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Retrying.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks, scenarios = [], [], [], []

        for _, pid, camid, trackid, sceneid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
            scenarios += [sceneid]

        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        scenarios = set(scenarios)

        num_pids = len(pids)

        num_cams = 1 + max(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        num_scenarios = len(scenarios)

        return num_pids, num_imgs, num_cams, num_views, num_scenarios

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views, _ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_query_views, _ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_views, _ = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

    def relabel(self, lists):
        relabeled = []
        pid_container = set()
        sceneid_container = set()
        for img_path, pid, camid, trackid, sceneid in lists:
            pid_container.add(pid)
            sceneid_container.add(sceneid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        sceneid2label = {sceneid: label for label, sceneid in enumerate(sceneid_container)}

        for img_path, pid, camid, trackid, sceneid in lists:
            pid = pid2label[pid]
            sceneid = sceneid2label[sceneid]
            relabeled.append([img_path, pid, camid, trackid, sceneid])

        return relabeled

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid, sceneid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, sceneid, img_path.split('/')[-1]

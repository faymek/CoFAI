import os
import os.path as osp
import glob
import statistics

from .bases import BaseImageDataset

class MuRI(BaseImageDataset):
    """
    MuRI Dataset
        muri/
          ├── train/
          │    ├── 0/
          │    │    ├── 0_8_155.jpg
          │    │    ├── 0_9_93_5_2.jpg
          │    ├── 1/
          │    │    ├── 1_3_25_...jpg
          │    └── ...
          ├── query/
          │    ├── 9/
          │    │    ├── 9_4_124.jpg
          │    ├── 14/
          │    │    ├── 14_14_3_92_173.jpg
          │    └── ...
          ├── gallery/
          │    ├── 9/
          │    │    ├── 9_3_59.jpg
          │    ├── 14/
          │    │    ├── 14_14_79_99_229.jpg
          │    └── ...
    """

    dataset_dir = "MuRI"

    def __init__(self, root='', verbose=True, **kwargs):

        super(MuRI, self).__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)

        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MuRI dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids, self.num_train_sceids \
            = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids, self.num_query_sceids \
            = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids, self.num_gallery_sceids \
            = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):

        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available.")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available.")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available.")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' is not available.")


    def _process_dir(self, dir_path, relabel=False):

        dataset = []
        pid_container = set()

        for pid_folder in os.listdir(dir_path):
            pid_path = osp.join(dir_path, pid_folder)
            if not osp.isdir(pid_path):
                continue

            try:
                pid = int(pid_folder)
            except ValueError:
                print(f" {pid_folder} NOT DIGITAL ID")
                continue

            pid_container.add(pid)
            img_names = os.listdir(pid_path)
            for img_name in img_names:
                if not img_name.lower().endswith(('.jpg','.jpeg','.png')):
                    continue

                img_path = osp.join(pid_path, img_name)


                camid = 0
                if 'query' in dir_path: camid = 0
                if 'gallery' in dir_path: camid = 1
                viewid = 0
                sceneid = 0

                dataset.append((img_path, pid, camid, viewid, sceneid))


        if relabel:
            unique_pids = sorted(list(pid_container))
            pid2label = {p: idx for idx, p in enumerate(unique_pids)}

            relabeled_dataset = []
            for (img_path, pid, camid, viewid, sceneid) in dataset:
                new_pid = pid2label[pid]
                relabeled_dataset.append((img_path, new_pid, camid, viewid, sceneid))
            dataset = relabeled_dataset

        return dataset

"""
VeRI‑Wild 2.0 Dataset Loader
----------------------------

VeRI-Wild2/
  ├── train_test_split/
  │    ├── train_list.txt
  │    ├── test_3000.txt
  │    ├── test_3000_query.txt
  │    ├── test_5000.txt
  │    ├── test_5000_query.txt
  │    ├── test_10000.txt
  │    └── test_10000_query.txt
  ├── images/
  │    ├── <pid>/<imgname>.jpg
  ├── test_split/
  │    ├── A_gallery.txt
  │    ├── A_query.txt
  │    ├── B_gallery.txt
  │    ├── B_query.txt
  │    ├── All_gallery.txt
  │    └── All_query.txt
  └── README.md
"""

import os
import os.path as osp
from .bases import BaseImageDataset


class VeRIWild2(BaseImageDataset):
    """
    VeRI‑Wild 2.0 子集加载：

      - train:      train_test_split/train_list.txt
      - query:      test_split/{A,B,All}_query.txt
      - gallery:    test_split/{A,B,All}_gallery.txt

    Usage:
        dataset = VeRIWild2(root='path/to/VeRI-Wild2', subset='All', verbose=True)
    """
    dataset_dir       = "VeRI-Wild2"
    train_list_file   = "train_test_split/train_list.txt"
    train_images_dir  = "images"
    test_split_dir_name = "test_split"
    subset_to_files = {
        "A": ("A_query.txt", "A_gallery.txt"),
        "B": ("B_query.txt", "B_gallery.txt"),
        "All": ("All_query.txt", "All_gallery.txt"),
    }

    def __init__(self, root='', subset='All', verbose=True, **kwargs):
        super().__init__()
        self.subset = self._normalize_subset(subset)
        self.dataset_dir      = osp.join(root, self.dataset_dir)
        self.train_list_path  = osp.join(self.dataset_dir, self.train_list_file)
        self.train_img_dir    = osp.join(self.dataset_dir, self.train_images_dir)
        self.test_split_dir   = osp.join(self.dataset_dir, self.test_split_dir_name)
        query_file, gallery_file = self.subset_to_files[self.subset]
        self.query_list_path  = osp.join(self.test_split_dir, query_file)
        self.gallery_list_path= osp.join(self.test_split_dir, gallery_file)

        self._check_before_run()

        train   = self._process_train(self.train_list_path, self.train_img_dir)
        query   = self._process_test(self.query_list_path,  self.dataset_dir)
        gallery = self._process_test(self.gallery_list_path,self.dataset_dir)



        if verbose:
            print(f"=> VeRI‑Wild 2.0 (test_{self.subset}) loaded")
            self.print_dataset_statistics(train, query, gallery)


        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids, self.num_train_sceids    = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids, self.num_query_sceids    = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids, self.num_gallery_sceids = \
            self.get_imagedata_info(self.gallery)

    @classmethod
    def _normalize_subset(cls, subset):
        if subset is None:
            return "All"
        subset = str(subset).strip()
        subset_map = {
            "a": "A",
            "b": "B",
            "all": "All",
            "A": "A",
            "B": "B",
            "All": "All",
        }
        if subset not in subset_map:
            raise ValueError(f"Unsupported VeRI-Wild2 subset '{subset}', expected one of A/B/All")
        return subset_map[subset]

    def _check_before_run(self):
        required = [
            self.dataset_dir,
            self.train_list_path,
            self.train_img_dir,
            self.test_split_dir,
            self.query_list_path,
            self.gallery_list_path,
        ]
        for path in required:
            if not osp.exists(path):
                raise RuntimeError(f"VeRI‑Wild 2.0 path dose not exist: {path}")

    def _process_train(self, list_path, img_root):

        dataset  = []
        pid_set  = set()

        with open(list_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for ln in lines:
            rel_path, pid_str, _ = ln.split()
            pid = int(pid_str)
            if pid != -1:
                pid_set.add(pid)

        pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_set))}

        for ln in lines:
            rel_path, pid_str, camid_str = ln.split()
            pid   = int(pid_str)
            camid = int(camid_str)

            if pid == -1:
                continue

            img_path = osp.join(img_root, rel_path)
            if not osp.exists(img_path):
                raise FileNotFoundError(img_path)

            new_pid = pid2label[pid]
            viewid  = 0
            sceneid = 0
            dataset.append((img_path, new_pid, camid, viewid, sceneid))

        return dataset

    def _process_test(self, list_path, root):

        dataset = []
        for l in open(list_path):
            l = l.strip()
            if not l:
                continue
            parts = l.split()
            if len(parts) != 2:
                raise ValueError(f"行格式错误: {l}")
            rel_path, pid_str = parts
            pid = int(pid_str)
            img_path = osp.join(root, rel_path)
            if not osp.exists(img_path):
                raise FileNotFoundError(img_path)

            if 'query' in list_path: camid = 0
            if 'gallery' in list_path: camid = 1
            viewid  = 0
            sceneid = 0
            dataset.append((img_path, pid, camid, viewid, sceneid))
        return dataset

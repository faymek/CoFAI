# -*- coding: utf-8 -*-
"""
VeRI‑Wild 1.0 Dataset Loader
----------------------------

目录组织（示例）：
VeRI-Wild/
  ├── images/
  │    ├── <pid>/<img>.jpg
  ├── train_test_split/
  │    ├── train_list.txt
  │    ├── test_3000_id.txt
  │    ├── test_3000_id_query.txt
  │    ├── test_5000_id.txt
  │    ├── test_5000_id_query.txt
  │    ├── test_10000_id.txt
  │    ├── test_10000_id_query.txt
  │    └── vehicle_info.txt
  └── README.md

文本行格式（以 test_10000_id_query.txt 为例）：
    31388/321972.jpg 31388 61
    <rel_path>       <pid>  <camid>

评测注意：按照官方说明，计算 mAP/CMC 时，需要把 **与 query 拍摄于相同 camera 且 pid 相同的 gallery 样本**排除。此加载器不做该过滤，请在评测阶段处理。
"""
import os
import os.path as osp
from .bases import BaseImageDataset

class VeRIWild(BaseImageDataset):
    """
    VeRI‑Wild 1.0

    Args:
        root (str): 数据根目录，包含 "VeRI-Wild"
        subset (str): 选择 "3000" | "5000" | "10000"
        verbose (bool)
        relabel (bool): 是否对训练集 pid 重新编号为连续 id（默认 True）
    """
    dataset_dir_name   = "VeRI-Wild"
    images_dir_name    = "images"
    split_dir_name     = "train_test_split"
    train_list_name    = "train_list.txt"
    query_tmpl         = "multi_test_{subset}_id_query.txt"
    gallery_tmpl       = "multi_test_{subset}_id.txt"

    def __init__(self, root='', subset='10000', verbose=True, relabel=True, **kwargs):
        super().__init__()
        self.subset = self._normalize_subset(subset)

        self.dataset_dir   = osp.join(root, self.dataset_dir_name)
        self.images_dir    = osp.join(self.dataset_dir, self.images_dir_name)
        self.split_dir     = osp.join(self.dataset_dir, self.split_dir_name)

        self.train_list    = osp.join(self.split_dir, self.train_list_name)
        self.query_list    = osp.join(self.split_dir, self.query_tmpl.format(subset=self.subset))
        self.gallery_list  = osp.join(self.split_dir, self.gallery_tmpl.format(subset=self.subset))

        self._check_before_run()

        train   = self._process_train(self.train_list, self.images_dir, relabel=relabel)
        query   = self._process_test(self.query_list,   self.images_dir)
        gallery = self._process_test(self.gallery_list, self.images_dir)

        if verbose:
            print(f"=> VeRI‑Wild 1.0 (test_{self.subset}) loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids,   self.num_train_imgs,   self.num_train_cams,   self.num_train_vids,   self.num_train_sceids   = self.get_imagedata_info(self.train)
        self.num_query_pids,   self.num_query_imgs,   self.num_query_cams,   self.num_query_vids,   self.num_query_sceids   = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids, self.num_gallery_sceids = self.get_imagedata_info(self.gallery)

    # ------------------------- helpers -------------------------
    @staticmethod
    def _normalize_subset(subset):
        if subset is None:
            return '10000'
        subset = str(subset).strip()
        if subset not in {'3000', '5000', '10000'}:
            raise ValueError(f"Unsupported VeRI-Wild subset '{subset}', expected one of 3000/5000/10000")
        return subset

    def _check_before_run(self):
        required = [
            self.dataset_dir,
            self.images_dir,
            self.split_dir,
            self.train_list,
            self.query_list,
            self.gallery_list,
        ]
        for p in required:
            if not osp.exists(p):
                raise RuntimeError(f"VeRI‑Wild path does not exist: {p}")

    def _process_train(self, list_path, img_root, relabel=True):

        dataset  = []
        pid_set  = set()

        with open(list_path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # 先收集有效 pid
        if relabel:
            for ln in lines:
                rel_path, pid_str, _ = ln.split()
                pid = int(pid_str)
                if pid != -1:
                    pid_set.add(pid)
            pid2label = {pid: i for i, pid in enumerate(sorted(pid_set))}

        for ln in lines:
            rel_path, pid_str, camid_str = ln.split()
            pid   = int(pid_str)
            camid = int(camid_str)

            if pid == -1:
                continue

            img_path = osp.join(img_root, rel_path)
            if not osp.exists(img_path):
                raise FileNotFoundError(img_path)

            new_pid = pid2label[pid] if relabel else pid
            viewid  = 0
            sceneid = 0
            dataset.append((img_path, new_pid, camid, viewid, sceneid))

        return dataset

    def _process_test(self, list_path, img_root):

        dataset = []
        with open(list_path, 'r') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                rel_path, pid_str, camid_str = ln.split()
                pid   = int(pid_str)
                camid = int(camid_str)

                if pid == -1:
                    continue

                img_path = osp.join(img_root, rel_path)
                if not osp.exists(img_path):
                    raise FileNotFoundError(img_path)

                viewid  = 0
                sceneid = 0
                dataset.append((img_path, pid, camid, viewid, sceneid))
        return dataset

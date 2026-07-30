from torch.utils.data import Sampler
from collections import defaultdict
import itertools
import random


class MultiViewSampler(Sampler):
    def __init__(self, data_source, batch_size: int, num_instances: int = 3):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances

        if self.batch_size % self.num_instances != 0:
            raise ValueError(
                "batch_size must be a multiple of num_instances."
                f" Got batch_size={batch_size}, num_instances={num_instances}"
            )
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, *_rest) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        for pid in self.index_dic:
            self.index_dic[pid].sort()
        self.pids = sorted(self.index_dic.keys())

        self.pid_to_chunks = defaultdict(list)
        for pid in self.pids:
            idxs = self.index_dic[pid]
            rng = random.Random(42)
            rng.shuffle(idxs)

            while len(idxs) % self.num_instances != 0:
                idxs.append(idxs[len(idxs) % len(idxs)])

            it = iter(idxs)
            self.pid_to_chunks[pid] = [
                list(itertools.islice(it, self.num_instances))
                for _ in range(len(idxs) // self.num_instances)
            ]
        self.length = sum(
            len(chunks) * self.num_instances for chunks in self.pid_to_chunks.values()
        )

    def __iter__(self):

        pid_to_chunks = {
            pid: chunks.copy() for pid, chunks in self.pid_to_chunks.items()
        }
        pid_pool = list(self.pids)
        output_idx = []

        while pid_pool:
            take = min(self.num_pids_per_batch, len(pid_pool))
            cur_pids = pid_pool[:take]
            del pid_pool[:take]

            for pid in cur_pids:
                chunk_list = pid_to_chunks[pid]
                if not chunk_list:
                    continue
                chunk = chunk_list.pop(0)
                output_idx.extend(chunk)
                if chunk_list:
                    pid_pool.append(pid)

        return iter(output_idx)

    def __len__(self):
        return self.length

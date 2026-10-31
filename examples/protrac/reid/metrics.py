import numpy as np
from torch.nn import functional as F


def retrieval_metrics(query, gallery, query_data, gallery_data):
    distance = (1 - F.normalize(query, dim=1) @ F.normalize(gallery, dim=1).t()).numpy()
    rankings = np.argsort(distance, axis=1)
    average_precision, rank1 = [], []
    for index, order in enumerate(rankings):
        same_id = gallery_data.pids[order] == query_data.pids[index]
        same_camera = gallery_data.cameras[order] == query_data.cameras[index]
        matches = same_id[~(same_id & same_camera)]
        if not matches.any():
            continue
        precision = matches.cumsum() / np.arange(1, len(matches) + 1)
        average_precision.append(float((precision * matches).sum() / matches.sum()))
        rank1.append(float(matches[0]))
    return {"mAP": 100 * float(np.mean(average_precision)), "Rank-1": 100 * float(np.mean(rank1))}

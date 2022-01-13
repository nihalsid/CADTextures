import torch
from tqdm import tqdm


def feature_stats(files):
    mean, std = {}, {}
    mean['input_positions'] = 0
    std['input_positions'] = 1
    mean['input_normals'] = 0
    std['input_normals'] = 1
    mean['input_laplacian'] = 0
    std['input_laplacian'] = 1
    mean['input_ff1'] = torch.zeros(1)
    std['input_ff1'] = torch.zeros(1)
    mean['input_ff2'] = torch.zeros(1)
    std['input_ff2'] = torch.zeros(1)
    mean['input_gcurv'] = torch.zeros(1)
    std['input_gcurv'] = torch.zeros(1)
    mean['input_mcurv'] = torch.zeros(1)
    std['input_mcurv'] = torch.zeros(1)
    for f in tqdm(files):
        pt_arxiv = torch.load(f)
        mean['input_ff1'] += pt_arxiv['input_ff1'].mean()
        std['input_ff1'] += (pt_arxiv['input_ff1'].std() ** 2)
        mean['input_ff2'] += pt_arxiv['input_ff2'].mean()
        std['input_ff2'] += (pt_arxiv['input_ff2'].std() ** 2)
        mean['input_gcurv'] += pt_arxiv['input_gcurv'].mean()
        std['input_gcurv'] += (pt_arxiv['input_gcurv'].std() ** 2)
        mean['input_mcurv'] += pt_arxiv['input_mcurv'].mean()
        std['input_mcurv'] += (pt_arxiv['input_mcurv'].std() ** 2)
    mean['input_ff1'] = mean['input_ff1'] / len(files)
    std['input_ff1'] = torch.sqrt(std['input_ff1'] / len(files))
    mean['input_ff2'] = mean['input_ff2'] / len(files)
    std['input_ff2'] = torch.sqrt(std['input_ff2'] / len(files))
    mean['input_gcurv'] = mean['input_gcurv'] / len(files)
    std['input_gcurv'] = torch.sqrt(std['input_gcurv'] / len(files))
    mean['input_mcurv'] = mean['input_mcurv'] / len(files)
    std['input_mcurv'] = torch.sqrt(std['input_mcurv'] / len(files))
    print(mean)
    print(std)
    torch.save({
        'mean': mean,
        'std': std
    }, "stats.pt")


if __name__ == '__main__':
    from pathlib import Path
    files = sorted(list(Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres_processed/").iterdir()))
    feature_stats(files)

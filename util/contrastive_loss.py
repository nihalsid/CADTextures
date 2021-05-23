import torch
import numpy as np
from scipy.linalg import block_diag


class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature, num_patch_x, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.num_patch_x = num_patch_x
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    @staticmethod
    def _get_correlated_mask(num_patch_x, batch_size):
        mask = torch.from_numpy(block_diag(*[np.ones((num_patch_x * num_patch_x, num_patch_x * num_patch_x)) for i in range(batch_size // (num_patch_x * num_patch_x))]))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        batch_size = zis.shape[0]
        similarity_matrix = self.similarity_function(zis, zjs)
        # filter out the scores from the positive samples
        positives = torch.diag(similarity_matrix).view(batch_size, 1)
        batch_mask = self._get_correlated_mask(self.num_patch_x, batch_size).type(torch.bool)
        negatives = similarity_matrix[batch_mask.cuda(zis.device)].view(batch_size, -1)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        labels = torch.zeros(batch_size).to(zis.device).long()
        loss = self.criterion(logits, labels)
        return loss / batch_size

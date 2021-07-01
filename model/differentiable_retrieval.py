import torch
from torch.autograd import Function
import torch.nn as nn


class RetrievalFunction(Function):

    @staticmethod
    def forward(ctx, similarity, candidates):
        ctx.save_for_backward(similarity, candidates)
        # selections = torch.einsum('ij,ijk->ik', similarity, candidates)
        selection_index = similarity.argmax(dim=1)
        selections = torch.einsum('ij,ijk->ik', nn.functional.one_hot(selection_index, num_classes=similarity.shape[1]).type(similarity.dtype), candidates)
        return selections

    @staticmethod
    def backward(ctx, grad_output):
        similarity, candidates = ctx.saved_tensors
        grad_similarity = grad_candidates = None
        if ctx.needs_input_grad[0]:
            grad_similarity = torch.einsum('ij,ijk->ik', grad_output, candidates.permute((0, 2, 1)))
        if ctx.needs_input_grad[1]:
            grad_candidates = torch.einsum('ij,ijk->ijk', similarity, grad_output.unsqueeze(1).repeat((1, similarity.shape[1], 1)))
        return grad_similarity, grad_candidates

import torch
import torch.nn as nn
from copy import deepcopy


class ANIL(nn.Module):
    def __init__(self, model, head_name="predictor",
                 inner_lr=1e-2, outer_lr=1e-3):

        super().__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.outer_opt = torch.optim.Adam(model.parameters(), lr=outer_lr)
        self.head_name = head_name

    def meta_step(self, task_batch):

        meta_loss = 0.0

        for support, query in task_batch:

            fast_model = deepcopy(self.model)

            head_params = [
                p for n, p in fast_model.named_parameters()
                if self.head_name in n
            ]

            support_loss = fast_model(*support)

            grads = torch.autograd.grad(
                support_loss,
                head_params,
                create_graph=True
            )

            for p, g in zip(head_params, grads):
                p.data -= self.inner_lr * g

            query_loss = fast_model(*query)
            meta_loss += query_loss

        meta_loss /= len(task_batch)

        self.outer_opt.zero_grad()
        meta_loss.backward()
        self.outer_opt.step()

        return meta_loss.item()

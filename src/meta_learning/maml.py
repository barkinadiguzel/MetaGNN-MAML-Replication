import torch
import torch.nn as nn
from copy import deepcopy


class MAML(nn.Module):
    def __init__(self, model, inner_lr=1e-2, outer_lr=1e-3):
        super().__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.outer_opt = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def inner_update(self, loss):
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=True
        )

        updated = []
        for p, g in zip(self.model.parameters(), grads):
            updated.append(p - self.inner_lr * g)

        return updated

    def meta_step(self, task_batch):
        meta_loss = 0.0

        for support, query in task_batch:

            fast_model = deepcopy(self.model)

            support_loss = fast_model(*support)
            fast_weights = torch.autograd.grad(
                support_loss,
                fast_model.parameters(),
                create_graph=True
            )

            for p, g in zip(fast_model.parameters(), fast_weights):
                p.data -= self.inner_lr * g

            query_loss = fast_model(*query)
            meta_loss += query_loss

        meta_loss /= len(task_batch)

        self.outer_opt.zero_grad()
        meta_loss.backward()
        self.outer_opt.step()

        return meta_loss.item()

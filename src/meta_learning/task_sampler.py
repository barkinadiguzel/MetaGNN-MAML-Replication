import random


class TaskSampler:

    def __init__(self, datasets, k_support=32, k_query=32):
        self.datasets = datasets
        self.k_support = k_support
        self.k_query = k_query

    def sample(self, batch_size):

        tasks = []

        for _ in range(batch_size):

            data = random.choice(self.datasets)
            random.shuffle(data)

            support = data[:self.k_support]
            query = data[self.k_support:
                         self.k_support + self.k_query]

            tasks.append((support, query))

        return tasks

# @Time   : 2020/9/23
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/9/23, 2020/12/28
# @Author : Zhen Tian, Yushuo Chen, Xingyu Pan
# @email  : chenyuwuxinn@gmail.com, chenyushuo@ruc.edu.cn, panxy@ruc.edu.cn

"""
recbole.data.dataloader.user_dataloader
################################################
"""
import torch
from logging import getLogger
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.interaction import Interaction
import numpy as np


class UserDataLoader(AbstractDataLoader):
    """:class:`UserDataLoader` will return a batch of data which only contains user-id when it is iterated.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        shuffle (bool): Whether the dataloader will be shuffle after a round.
            However, in :class:`UserDataLoader`, it's guaranteed to be ``True``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        if shuffle is False:
            shuffle = True
            self.logger.warning("UserDataLoader must shuffle the data.")

        self.uid_field = dataset.uid_field
        self.user_list = Interaction({self.uid_field: torch.arange(dataset.user_num)})
        self.sample_size = len(self.user_list)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    def collate_fn(self, index):
        index = np.array(index)
        return self.user_list[index]


class GSAUDataLoader:
    def __init__(self, graph_dataloader, sequential_dataloader):
        self.logger = getLogger()
        self.graph_dataloader = graph_dataloader
        self.sequential_dataloader = sequential_dataloader

        # Create iterators for both dataloaders
        self.graph_iter = iter(self.graph_dataloader)
        self.sequential_iter = iter(self.sequential_dataloader)

        self.a_task_done = False
    
    def __len__(self):
        return max(len(self.graph_dataloader), len(self.sequential_dataloader))

    def __iter__(self):
        # Reset the iterators at the begining of each epoch
        self.graph_iter = iter(self.graph_dataloader)
        self.sequential_iter = iter(self.sequential_dataloader)
        self.a_task_done = False
        return self

    def __next__(self):
        """Fetch the next batch of data for both encoders."""
        try:
            graph_batch = next(self.graph_iter)
        except StopIteration:
            if not self.a_task_done: # This task is done first
                self.graph_iter = iter(self.graph_dataloader)
                graph_batch = next(self.graph_iter)
                self.a_task_done = True
            else: # The other task is already done
                raise StopIteration  # Stop when both tasks are done

        try:
            sequential_batch = next(self.sequential_iter)
        except StopIteration:
            if not self.a_task_done: # This task is done first
                self.sequential_iter = iter(self.sequential_dataloader)
                sequential_batch = next(self.sequential_iter)
                self.a_task_done = True
            else: # The other task is already done
                raise StopIteration  # Stop when both tasks are done
            
        # Combine the batches from task1 and task2
        return {
            'graph_batch': graph_batch,
            'sequential_batch': sequential_batch
        }

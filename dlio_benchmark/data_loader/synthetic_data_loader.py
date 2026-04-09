"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import math
import numpy as np
import os
import time
import torch

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.utils.utility import utcnow, Profile, dft_ai

dlp = Profile(MODULE_DATA_LOADER)
ITER_TIME = float(os.environ.get('DLIO_SLEEP_TIME', 1))
logging.getLogger("SyntheticDataLoader").info(f"DLIO_SLEEP_TIME={ITER_TIME}")

class SyntheticDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.SYNTHETIC)
        shape = self._args.resized_image.shape
        # Calculate local samples for this rank
        total_samples = self.num_samples
        samples_per_proc = int(math.ceil(total_samples / self._args.comm_size))
        start_sample = self._args.my_rank * samples_per_proc
        end_sample = (self._args.my_rank + 1) * samples_per_proc - 1
        if end_sample > total_samples - 1:
            end_sample = total_samples - 1
        local_num_samples = end_sample - start_sample + 1
        
        # Calculate number of batches
        self.num_batches = int(math.ceil(local_num_samples / self.batch_size))
        
        # Pre-create a batch of zeros
        # Use pinned torch tensor instead, and expand to 4D to avoid overhead in validate_data
        # Original logic in validate_data: data.unsqueeze(1).repeat(1, 3, 1, 1)
        # This converts (N, H, W) -> (N, 3, H, W)
        self.zero_batch = torch.zeros((self.batch_size, 3, shape[0], shape[1]), dtype=torch.uint8).pin_memory()

    @dlp.log
    def read(self, init=False):
        return
    
    @dft_ai.data.item
    def getitem(self):
        return self.batch

    @dlp.log
    def next(self):
        super().next()
<<<<<<< HEAD
        self.logger.debug(f"{utcnow()} Iterating pipelines by {self._args.my_rank} rank ")
        self.read(True)

        step = 1
        dft_ai.dataloader.fetch.start()
        while step < self.num_samples // self.batch_size:
            dft_ai.dataloader.fetch.stop()
            dft_ai.update(step=step)
            step += 1
            yield self.getitem()
            dft_ai.dataloader.fetch.start()

        self.epoch_number += 1
        dft_ai.update(epoch=self.epoch_number)
=======
        for step in dlp.iter(range(self.num_batches)):
            dlp.update(step=step)
            time.sleep(ITER_TIME)
            yield self.zero_batch
>>>>>>> cd74d4e (added all changes)

    @dlp.log
    def finalize(self):
        return

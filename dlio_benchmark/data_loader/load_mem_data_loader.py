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
import logging
import math
import numpy as np
import os
import time

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.data_loader.torch_data_loader import TorchDataLoader
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_LOADER)
ITER_TIME = float(os.environ.get('DLIO_SLEEP_TIME', 1))

class LoadMemDataLoader(BaseDataLoader):
    """
    Load Memory Data Loader - Preloads entire dataset into memory for fast access.
    
    This data loader uses PyTorch's DataLoader internally to read all data once,
    stores it in memory, and then serves batches from the preloaded cache. This
    eliminates I/O overhead for subsequent epochs and iterations.
    
    Architecture:
    - Uses TorchDataLoader internally for the initial data loading
    - Preloads all batches into a list during the read() phase
    - Serves batches from memory during next() iterations
    
    Use Cases:
    - Benchmarking pure compute performance without I/O interference
    - Small to medium datasets that fit in available RAM
    - Multi-epoch training where data can be cached
    - Scenarios where I/O latency needs to be eliminated
    
    Memory Requirements:
    - Requires enough RAM to hold the entire dataset for this rank
    - Memory usage = (num_samples / num_ranks) * sample_size
    
    Example Configuration:
        ++workload.reader.data_loader=load_mem
        ++workload.dataset.format=jpeg  # or any supported format
        ++workload.reader.batch_size=32
    
    Note: The first epoch will include I/O time for loading data into memory.
    Subsequent epochs (if any) will be purely from memory.
    """
    
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.LOAD_MEM)
        
        # Create internal PyTorch dataloader
        self._torch_loader = TorchDataLoader(format_type, dataset_type, epoch)
        
        # Storage for preloaded data
        self._preloaded_batches = []
        self._preloaded = False

    @dlp.log
    def read(self, init=False):
        """Preload all data into memory using the PyTorch dataloader"""
        if self._preloaded:
            return
            
        self._torch_loader.read()
        # Use the torch loader to read all batches into memory

        self.logger.info("Preloading data")
        batch_num = 0
        import torch
        for batch in self._torch_loader.next():
            # Preprocess the batch to avoid overhead during benchmark loop
            # Unpack batch (it might be tuple or just input)
            if isinstance(batch, (tuple, list)):
                input_data = batch[0]
                target = batch[1] if len(batch) > 1 else None
            else:
                input_data = batch
                target = None

            if isinstance(input_data, torch.Tensor):
                # Expand channels if needed (N, H, W) -> (N, 3, H, W)
                if len(input_data.shape) == 3:
                     input_data = input_data.unsqueeze(1).repeat(1, 3, 1, 1)
                
                # Ensure it is pinned for faster transfer
                if not input_data.is_pinned():
                    input_data = input_data.pin_memory()
                
                # Re-pack batch
                if isinstance(batch, (tuple, list)):
                    batch = (input_data, *batch[1:])
                else:
                    batch = input_data

            self.logger.info(f"Rank {self._args.my_rank} Loaded batch {batch_num}")
            self._preloaded_batches.append(batch)
            batch_num += 1
        
        self._preloaded = True
        
        if self._args.my_rank == 0:
            self.logger.info(f"Preloaded {len(self._preloaded_batches)} batches into memory")

    @dlp.log
    def next(self):
        """Yield batches from preloaded data"""
        super().next()
        
        # Ensure data is preloaded
        if not self._preloaded:
            self.read(True)
        
        # Yield from preloaded batches
        for step, batch in dlp.iter(enumerate(self._preloaded_batches, 1)):
            dlp.update(step=step)
            time.sleep(ITER_TIME)
            yield batch

    @dlp.log
    def finalize(self):
        """Clean up resources"""
        self._torch_loader.finalize()
        # Note: We intentionally do NOT clear _preloaded_batches here
        # The purpose of load_mem is to keep data in memory across epochs
        return

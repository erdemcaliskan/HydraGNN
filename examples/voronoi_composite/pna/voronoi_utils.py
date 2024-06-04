import json
import os
import os.path as osp
from typing import Callable, List, Optional

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)
from torch_geometric.io import read_txt_array

import torch_geometric.transforms as T

import glob
import re

class VoronoiNet(InMemoryDataset):

    def __init__(
        self,
        root: str,
        split: str = 'trainval',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        
        self.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['processed_data.pt']
    
    def process(self) -> None:

        def extract_numbers(filepath):
            match_sve = re.search(r'SVE(\d+)', filepath)
            match_v = re.search(r'v(\d+)', filepath)
            match_m = re.search(r'm(\d+)', filepath)
            sve_number = int(match_sve.group(1)) if match_sve else 0
            v_number = int(match_v.group(1)) if match_v else 0
            m_number = int(match_m.group(1)) if match_m else 0
            return (v_number, sve_number, m_number)
        
        data_list = []

        v_paths = glob.glob("../dataset/*")
        sorted_v_paths = sorted(v_paths, key=extract_numbers)

        for v in sorted_v_paths:
            m_paths = glob.glob(v + "/*")
            sorted_m_paths = sorted(m_paths, key=extract_numbers)
            for m in sorted_m_paths:
                results_path = m + '/AllFld_SVE.csv'
                tensor_y = read_txt_array(results_path,sep=',')
                SVE_paths = m + '/SVE*.csv'
                file_paths = glob.glob(SVE_paths)

                sorted_file_paths = sorted(file_paths, key=extract_numbers)
                for i,name in enumerate(sorted_file_paths):
                    tensor = read_txt_array(name,sep=',')
                    tensor = tensor.view(-1, tensor.shape[-1])
                    pos = tensor[:, :2]
                    x = tensor[:, 2:]
                    y = tensor_y[i, 5:] if len(tensor_y.shape) == 2 else tensor_y[5:]
                    #pos = tensor[:, :2]
                    #x = tensor[:, 2:]
                    #y = tensor_y[i, -1] #.type(torch.long)
                    data = Data(pos=pos, x=x, y=y)

                    #pre_transform=T.KNNGraph(k=5)
                    #pre_transform=T.KNNGraph(k=5,force_undirected=True)
                    pre_transform = T.RadiusGraph(r=5, loop=False, max_num_neighbors=64)
                    data = pre_transform(data)
                    distance = T.Distance(norm=False)
                    local_cartesian = T.LocalCartesian(norm=False)
                    data = distance(data)
                    data = local_cartesian(data)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
                self.save(data_list,'processed/processed_data.pt')

#    def len(self):
#        return len(self.dataset)
#
#    def get(self, idx):
#        return self.dataset[idx]

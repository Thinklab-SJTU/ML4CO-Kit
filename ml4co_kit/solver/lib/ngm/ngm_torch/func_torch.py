r"""
Pytorch implementation of NGM
"""
# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import torch
from ml4co_kit.utils.file_utils import download
from ml4co_kit.solver.lib.ngm.ngm_torch.utils import _load_model, _check_and_init_gm 
from ml4co_kit.solver.lib.ngm.ngm_torch.model import NGM_Net

ngm_pretrain_path = {
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/ngm_voc_pytorch.pt',
             'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/pytorch_backend/ngm_voc_pytorch.pt',
             'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1POvy6J-9UDNy93qJCKu-czh2FCYkykMK'],
             '60dbc7cc882fd88de4fc9596b7fb0f4a'),
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/ngm_willow_pytorch.pt',
                'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/pytorch_backend/ngm_willow_pytorch.pt',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1ZdlUxyeNoIjA74QTr5wxwQ-vBrr2MBaL'],
                'dd13498bb385df07ac8530da87b14cd6'),
}

def ngm_torch(K, n1, n2, n1max, n2max, x0, gnn_channels, sk_emb, sk_max_iter, sk_tau, network, pretrain):
    """
    Pytorch implementation of NGM
    """
    if K is None:
        forward_pass = False
        device = torch.device('cpu')
    else:
        forward_pass = True
        device = K.device
    if network is None:
        network = NGM_Net(gnn_channels, sk_emb)
        network = network.to(device)
        if pretrain:
            if pretrain in ngm_pretrain_path:
                url, md5 = ngm_pretrain_path[pretrain]
                try:
                    filename = download(f'ngm_{pretrain}_pytorch.pt', url, md5)
                except:
                    filename = os.path.dirname(__file__) + f'/temp/ngm_{pretrain}_pytorch.pt'
                filename="/home/zhanghang/caibinghao/ML4CO-Kit/ngm_voc_pytorch.pt"
                _load_model(network, filename, device)
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {ngm_pretrain_path.keys()}')

    if forward_pass:
        batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
        v0 = v0 / torch.mean(v0)
        result = network(K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


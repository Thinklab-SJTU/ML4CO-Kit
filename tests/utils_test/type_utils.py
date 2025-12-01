r"""
Test Type Utils Module.
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


import torch
import numpy as np
from ml4co_kit import to_numpy, to_tensor


class TypeUtilsTester(object):
    """Test cases for file utility functions."""
    
    def __init__(self) -> None:
        pass
    
    def test(self):
        # Generate data to tested
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        list = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test to_numpy
        assert np.array_equal(to_numpy(array), array)
        assert np.array_equal(to_numpy(tensor), array)
        assert np.array_equal(to_numpy(list), array)
        
        # Test to_tensor
        assert torch.allclose(to_tensor(array), tensor)
        assert torch.allclose(to_tensor(tensor), tensor)
        assert torch.allclose(to_tensor(list), tensor)
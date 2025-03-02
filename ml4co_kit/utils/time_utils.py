r"""
The utilities used to show progress bar and running time.
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


import time
from tqdm import tqdm
from typing import Callable, Iterable, TypeVar


T = TypeVar('T')


def iterative_execution(
    func: Callable[[Iterable[T]], Iterable[T]], 
    iterable: Iterable[T], 
    desc: str = "Running", 
    show_time: bool = False
) -> Iterable[T]:
    r"""
    Applies a function to an iterable and optionally displays a progress bar using tqdm.    
    :param func: iterable object, the function to display a progress bar.
    :param iterable: iterable object, the parameter applied to 'func'.
    :param desc: string, the descriptive text for the progress bar. Defaults to Running.
    :param show_time: boolean, whether to display a progress bar.
    """
    if show_time:
        return tqdm(func(iterable), desc=desc)
    else:
        return func(iterable)
    

def iterative_execution_for_file(
    iterable: Iterable[T], 
    desc: str = "Running", 
    show_time: bool = False
) -> Iterable[T]:
    r"""
    Conditionally wraps an iterable with tqdm to display a progress bar.
    :param iterable: iterable object, the input iterable to be processed.
    :param desc: string, the descriptive text for the progress bar. Defaults to Running.
    :param show_time: boolean, whether to display a progress bar.
    """
    if show_time:
        return tqdm(iterable, desc=desc)
    else:
        return iterable
    

class Timer(object):
    r"""
    A utility class for measuring the execution time of code blocks.
    :param apply: boolean, whether to apply the timer on the code blocks.
    :param start_time: float, the time to mark the start time of code blocks.
    :param end_time: float, the time to mark the end time of code blocks.
    :param use_time: float, the time code blocks use.
    """
    def __init__(self, apply: bool = True):
        self.apply = apply
        self.start_time = None
        self.end_time = None
        self.use_time = None
        
    def start(self):
        r"""
        mark a start time.
        """
        if self.apply:
            self.start_time = time.time()
    
    def end(self):
        r"""
        mark an end time and calcuate the used time.
        """
        if self.apply:
            self.end_time = time.time()
            self.use_time = self.end_time - self.start_time
    
    def show_time(self):
        r"""
        show the using time.
        """
        if self.apply:
            print(f"Using Time: {self.use_time}")
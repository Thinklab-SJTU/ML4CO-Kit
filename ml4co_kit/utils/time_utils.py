import time
from tqdm import tqdm
from typing import Callable, Iterable, TypeVar


T = TypeVar('T')


def iterative_execution(
    func: Callable[[Iterable[T]], Iterable[T]], 
    iterable: Iterable[T], 
    desc: str = "Runing", 
    show_time: bool = False
) -> Iterable[T]:
    if show_time:
        return tqdm(func(iterable), desc=desc)
    else:
        return func(iterable)
    

def iterative_execution_for_file(
    iterable: Iterable[T], 
    desc: str = "Runing", 
    show_time: bool = False
) -> Iterable[T]:
    if show_time:
        return tqdm(iterable, desc=desc)
    else:
        return iterable
    

class Timer(object):
    def __init__(self, apply: bool = True):
        self.apply = apply
        self.start_time = None
        self.end_time = None
        self.use_time = None
        
    def start(self):
        if self.apply:
            self.start_time = time.time()
    
    def end(self):
        if self.apply:
            self.end_time = time.time()
            self.use_time = self.end_time - self.start_time
    
    def show_time(self):
        if self.apply:
            print(f"Using Time: {self.use_time}")
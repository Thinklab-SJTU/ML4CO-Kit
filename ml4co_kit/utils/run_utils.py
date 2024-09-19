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
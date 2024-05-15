class BaseEnv:
    def __init__(
        self,
        name: str,
        mode: str = "train",
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: int = 0,
        device: str = "cpu"
    ):
        self.name = name
        self.mode = mode
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.device = device
        
    def load_data(self):
        raise NotImplementedError(
            "``load_data`` is required to implemented in subclasses."
        )

    def train_dataloader(self):
        raise NotImplementedError(
            "``train_dataloader`` is required to implemented in subclasses."
        )

    def val_dataloader(self):
        raise NotImplementedError(
            "``val_dataloader`` is required to implemented in subclasses."
        )

    def test_dataloader(self):
        raise NotImplementedError(
            "``test_dataloader`` is required to implemented in subclasses."
        )

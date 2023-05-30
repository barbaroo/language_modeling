class MLMTrainerParams:
    def __init__(self, dataset_path, batch_size, lr, epochs, device, n_examples, split, mask_probability):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.n_examples = n_examples
        self.split = split
        self.mask_probability = mask_probability

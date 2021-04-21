RANDOM_STATE = 64


class OptimizerParams:
    def __init__(self) -> None:
        self.lr = 1e-2
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 0
        self.amsgrad = True


class SchedulerPrams:
    def __init__(self) -> None:
        self.min_lr = 1e-7
        self.verbose = True
        self.mode = "min"
        self.patience = 5
        self.threshold = 1e-3


class TrainParams:
    def __init__(self) -> None:
        self.train_size = 0.85
        self.num_landmarks = 971
        self.dropout_prob = 0.2
        self.check_val_every_n_epoch = 5
        self.flush_logs_every_n_steps = 100
        # 16 or 32
        self.precision = 16
        self.max_epochs = 50
        self.train_backbone = True
        self.img_size_in_batch = (256, 256)


class LossParams:
    def __init__(self) -> None:
        self.w = 0.2
        self.eps = 2
        self.redcution = "sum"

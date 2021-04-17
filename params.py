RANDOM_STATE = 123


class OptimizerParams:
    def __init__(self) -> None:
        self.lr = 1e-3
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 0
        self.amsgrad = False


class TrainParams:
    def __init__(self) -> None:
        self.train_size = 0.8
        self.num_landmarks = 971
        self.dropout_prob = 0.5
        self.check_val_every_n_epoch = 2
        self.flush_logs_every_n_steps = 100
        # 16 or 32
        self.precision = 16
        self.max_epochs = 5


class LossParams:
    def __init__(self) -> None:
        self.w = 0.5
        self.eps = 2
        self.redcution = "sum"

RANDOM_STATE = 32


class OptimizerParams:
    def __init__(self) -> None:
        self.lr = 1e-1
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 0
        self.amsgrad = True


class SchedulerPrams:
    def __init__(self) -> None:
        self.T_max = 1000
        self.eta_min = 1e-4


class TrainParams:
    def __init__(self) -> None:
        self.train_size = 0.99
        self.num_landmarks = 971
        self.dropout_prob = 0.1
        self.check_val_every_n_epoch = 1
        self.flush_logs_every_n_steps = 100
        # 16 or 32
        self.precision = 16
        self.max_epochs = 50
        self.train_backbone = True
        self.img_size_in_batch = (256, 256)
        self.train_full_model_after_epoch = 8


class LossParams:
    def __init__(self) -> None:
        self.w = 8
        self.eps = 2
        self.redcution = "sum"

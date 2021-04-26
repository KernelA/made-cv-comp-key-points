RANDOM_STATE = 64


class OptimizerParams:
    def __init__(self) -> None:
        self.lr = 1e-3
        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 0
        self.amsgrad = True


class SchedulerPrams:
    def __init__(self) -> None:
        self.base_lr = 1e-4
        self.max_lr = 1e-2
        self.step_size_up = 250
        self.mode = "triangular2"
        self.cycle_momentum = False


class TrainParams:
    def __init__(self) -> None:
        self.train_size = 0.99
        self.num_landmarks = 971
        self.dropout_prob = 0.25
        self.check_val_every_n_epoch = 1
        self.flush_logs_every_n_steps = 100
        # 16 or 32
        self.precision = 32
        self.max_epochs = 50
        self.train_backbone = True
        self.img_size_in_batch = (256, 256)


class LossParams:
    def __init__(self) -> None:
        self.w = 8
        self.eps = 2
        self.redcution = "sum"

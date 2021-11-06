from aydin.util.log.log import lprint


class CatBoostStopTrainingCallback:
    def __init__(self):
        self.continue_training = True

    def after_iteration(self, info):
        lprint(
            f"{info.iteration}: learn: {info.metrics['learn']['MAE'][-1]}  test: {info.metrics['learn']['MAE'][-1]}"
        )
        return self.continue_training

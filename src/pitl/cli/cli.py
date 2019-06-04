import click

from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.lgbm import LightGBMRegressor


scales = [1, 3, 5, 11, 21, 23, 47, 95]
widths = [3, 3, 3,  3,  3,  3,  3,  3]


def train(widths=widths, scales=scales):
    generator = MultiscaleConvolutionalFeatures(kernel_widths=widths,
                                                kernel_scales=scales,
                                                exclude_center=False
                                                )

    regressor = LightGBMRegressor(num_leaves=64,
                                  max_depth=7,
                                  n_estimators=1024,
                                  learning_rate=0.01,
                                  eval_metric='l1',
                                  early_stopping_rounds=None)

    it = ImageTranslator(generator, regressor)


@click.command()
def pitl_cli():
    train()


if __name__ == '__main__':
    pitl_cli()

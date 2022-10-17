from torch import nn
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from aydin.nn.datasets.grid_masked_dataset import GridMaskedDataset
from aydin.util.torch.device import get_torch_device


def n2s_train(
    input_image,
    model: nn.Module,
    *,
    nb_epochs: int = 128,
    lr: float = 0.001,
    # patch_size: int = 32,
    patience: int = 128,
    verbose: bool = True,
):
    """
    Noise2Self training method.

    Parameters
    ----------
    input_image
    model : nn.Module
    nb_epochs : int
    lr : float
    patience : int
    verbose : bool

    """
    device = get_torch_device()

    torch.autograd.set_detect_anomaly(True)

    model = model.to(device)
    print(f"device {device}")

    optimizer = AdamW(model.parameters(), lr=lr)

    # optimizer = ESAdam(
    #     chain(model.parameters()),
    #     lr=learning_rate,
    #     start_noise_level=0.001,
    #     weight_decay=1e-9,
    # )

    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        verbose=True,
        patience=patience // 8,
    )

    loss_function1 = MSELoss()

    dataset = GridMaskedDataset(input_image)
    print(f"dataset length: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=16, num_workers=3, shuffle=False)

    model.train()

    for epoch in range(nb_epochs):
        loss = 0
        for i, batch in enumerate(data_loader):
            original_patch, net_input, mask = batch

            original_patch = original_patch.to(device)
            net_input = net_input.to(device)
            mask = mask.to(device)

            net_output = model(net_input)

            if epoch == 25555:
                import napari

                viewer = napari.Viewer()
                viewer.add_image(
                    model(original_patch.to(device)).detach().cpu().numpy(),
                    name=f"{epoch}",
                )
                napari.run()

            loss = loss_function1(net_output * mask, original_patch * mask)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        scheduler.step(loss)

        if verbose:
            print("Loss (", epoch, "): \t", round(loss.item(), 8))

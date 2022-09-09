import numpy
from torch.utils.data import Dataset

from aydin.util.array.nd import extract_tiles


class TorchDataset(Dataset):
    def __init__(self, input_image, target_image, tilesize, self_supervised=False):
        """ """

        num_channels_input = input_image.shape[1]
        num_channels_target = target_image.shape[1]

        def extract(image):
            return extract_tiles(
                image, tile_size=tilesize, extraction_step=tilesize, flatten=True
            )

        bc_flat_input_image = input_image.reshape(-1, *input_image.shape[2:])
        bc_flat_input_tiles = numpy.concatenate(
            [extract(x) for x in bc_flat_input_image]
        )
        self.input_tiles = bc_flat_input_tiles.reshape(
            -1, num_channels_input, *bc_flat_input_tiles.shape[1:]
        )

        if self_supervised:
            self.target_tiles = self.input_tiles
        else:
            bc_flat_target_image = target_image.reshape(-1, *target_image.shape[2:])
            bc_flat_target_tiles = numpy.concatenate(
                [extract(x) for x in bc_flat_target_image]
            )
            self.target_tiles = bc_flat_target_tiles.reshape(
                -1, num_channels_target, *bc_flat_target_tiles.shape[1:]
            )

        mask_image = numpy.zeros_like(input_image)
        # mask_image[validation_voxels] = 1

        bc_flat_mask_image = mask_image.reshape(-1, *mask_image.shape[2:])
        bc_flat_mask_tiles = numpy.concatenate([extract(x) for x in bc_flat_mask_image])
        self.mask_tiles = bc_flat_mask_tiles.reshape(
            -1, num_channels_input, *bc_flat_mask_tiles.shape[1:]
        )

    def __len__(self):
        return len(self.input_tiles)

    def __getitem__(self, index):
        input = self.input_tiles[index, ...]
        target = self.target_tiles[index, ...]
        mask = self.mask_tiles[index, ...]

        return (input, target, mask)

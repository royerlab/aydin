"""Data model managing loaded images, file paths, and their transformations."""

from copy import deepcopy
from dataclasses import dataclass
from os import listdir
from os.path import isdir, isfile, join
from pathlib import Path

from aydin.io import imread
from aydin.io.utils import hyperstack_arrays, split_image_channels
from aydin.util.log.log import aprint


@dataclass
class ImageRecord:
    """A single image entry in the data model.

    Attributes
    ----------
    filename : str
        Display name of the image file.
    array : object
        The image data as a numpy ndarray.
    metadata : object
        A ``FileMetadata`` instance with axes, shape, and dtype info.
    denoise : bool
        Whether this image is marked for denoising.
    filepath : str
        Full file path on disk.
    output_folder : str
        Directory where denoised output will be saved.
    """

    filename: str
    array: object
    metadata: object
    denoise: bool
    filepath: str
    output_folder: str


class DataModel:
    """Manages the image data model for the Aydin Studio GUI.

    Handles file path storage, image loading, channel splitting,
    hyperstacking, and coordinates updates between the data layer
    and the GUI tab views.

    Parameters
    ----------
    parent : MainPage
        The parent MainPage widget that owns the tab views.
    """

    def __init__(self, parent=None):
        """Initialize the data model with empty file paths and image lists.

        Parameters
        ----------
        parent : MainPage, optional
            The parent MainPage widget that owns the tab views.
        """
        self.parent = parent
        self._filepaths = dict()
        self._images = []

    @property
    def filepaths(self):
        """File paths loaded into the model with their image data.

        Returns
        -------
        dict
            Mapping of file path strings to ``(array, metadata)`` tuples
            where ``array`` is a numpy ndarray and ``metadata`` is a
            ``FileMetadata`` instance.
        """
        return self._filepaths

    def clear_filepaths(self):
        """Clear all file paths and images from the model.

        Resets the internal file path dictionary, triggers updates on the
        files tab view, and clears all loaded images.
        """
        self._filepaths = dict()
        self.update_files_tabview()
        self.clear_images()

    def add_filepaths(self, file_paths):
        """Add file paths to the model if not already present.

        Reads each file, stores the (array, metadata) pair, and triggers
        updates to the files and images tab views. Directories are expanded
        to include their contained files.

        Parameters
        ----------
        file_paths : list of str
            File paths or directory paths to add.
        """
        new_paths_added = False
        new_images = {}

        # If we drag and drop folders we add the files in the folders instead:
        if all(isdir(folder_path) for folder_path in file_paths):
            expanded_paths = []
            for folder_path in file_paths:
                for file_path in listdir(folder_path):
                    expanded_paths.append(join(folder_path, file_path))
            file_paths = expanded_paths

        # Add each path checking whether it is already present and an actual file:
        for file_path in file_paths:
            if file_path not in self._filepaths and isfile(file_path):
                array, metadata = imread(file_path)
                if array is None and metadata is None:
                    continue

                new_paths_added = True
                self._filepaths[file_path] = (array, metadata)
                new_images[file_path] = (array, metadata)

        if new_paths_added:
            self.update_files_tabview()
            self.add_images(new_images)

    def remove_filepaths(self, fpaths):
        """Remove the given file paths and their corresponding images from the model.

        Parameters
        ----------
        fpaths : list of str
            File paths to remove from the model.
        """
        for fpath in fpaths:
            del self._filepaths[fpath]
            self.remove_image(fpath)

    @property
    def images(self):
        """List of image records currently in the model.

        Returns
        -------
        list of ImageRecord
            List of :class:`ImageRecord` dataclass instances.
        """
        return self._images

    def clear_images(self):
        """Clear all images from the model and update the images tab view."""
        self._images.clear()
        self.update_images_tabview()

    def add_images(self, new_images, denoise=True):
        """Add images to the model and update the images tab view.

        Parameters
        ----------
        new_images : dict
            Mapping of file path strings to ``(array, metadata)`` tuples
            where ``array`` is a numpy ndarray and ``metadata`` is a
            ``FileMetadata`` instance.
        denoise : bool, optional
            Whether to mark the images for denoising. Default is True.
        """
        for path, (array, metadata) in new_images.items():
            self._images.append(
                ImageRecord(
                    filename=Path(path).name,
                    array=array,
                    metadata=metadata,
                    denoise=denoise,
                    filepath=path,
                    output_folder=str(Path(path).resolve().parent),
                )
            )

        self.update_images_tabview()

    def remove_image(self, image_filepath):
        """Remove all images associated with the given file path from the model.

        Parameters
        ----------
        image_filepath : str
            File path whose associated images should be removed.
        """
        indices2remove = []
        for idx, imagelist_item in enumerate(self._images):
            if (
                Path(image_filepath).parents[0]
                == Path(imagelist_item.filepath).parents[0]
                and Path(image_filepath).name in imagelist_item.filename
            ):
                indices2remove.append(idx)

        for idx in reversed(indices2remove):
            self._images.pop(idx)

        self.update_images_tabview()

    @property
    def images_to_denoise(self):
        """Subset of images that are marked for denoising.

        Returns
        -------
        list
            Image records where the denoise flag is True.
        """
        return list(filter(lambda image: image.denoise, self.images))

    def set_image_to_denoise(self, filename, new_value):
        """Mark or unmark an image for denoising by its filename.

        Updates the denoise flag and triggers dimension and cropping tab
        refreshes.

        Parameters
        ----------
        filename : str
            The display name of the image to update.
        new_value : bool
            True to mark the image for denoising, False to unmark it.
        """
        for imagelist_item in self._images:
            if imagelist_item.filename == filename:
                imagelist_item.denoise = new_value
                self.update_dimensions_tabview()
                self.update_cropping_tabview()

    def update_image_output_folder(self, filename, new_value):
        """Update the output folder for an image identified by filename.

        Parameters
        ----------
        filename : str
            The display name of the image to update.
        new_value : str
            The new output folder path.
        """
        for imagelist_item in self._images:
            if imagelist_item.filename == filename:
                imagelist_item.output_folder = new_value

    def set_split_channels(self, filename, filepath, new_value: bool):
        """Split or re-merge channels of an image by its filename.

        When ``new_value`` is True, splits the image along its channel axis
        into separate single-channel images. When False, re-merges previously
        split channels back into the original multi-channel image.

        Parameters
        ----------
        filename : str
            The display name of the image.
        filepath : str
            The full file path of the image.
        new_value : bool
            True to split channels, False to re-merge.

        Returns
        -------
        int or None
            Returns -1 if the image has no channel axis. Returns None
            otherwise.
        """
        # Fetch the self._images elements that are associated by their file names
        imagelist_items = [elem for elem in self._images if filename in elem.filename]

        if new_value:  # if the split checkbox value is True, we try to split

            # There should be only one related image as a non-splitted file
            if len(imagelist_items) != 1:
                return

            imagelist_item = imagelist_items[0]

            if imagelist_item.metadata.axes.find("C") == -1:
                aprint("Array has no channel axis detected")
                return -1

            self.remove_image(filepath)  # Remove the existing image from the model

            # Split the array and collect new arrays and metadatas
            splitted_arrays, metadatas = split_image_channels(
                imagelist_item.array, deepcopy(imagelist_item.metadata)
            )

            # Prepare new file names for the splitted images
            filenames = [
                f"channel_{ch_idx}_{filename}" for ch_idx in range(len(splitted_arrays))
            ]

            # Add the splitted images to the model
            new_images = {}
            for filename, array, metadata in zip(filenames, splitted_arrays, metadatas):
                new_images[str(Path(filepath).with_name(filename))] = (
                    array,
                    metadata,
                )
            self.add_images(new_images)

            self.update_images_tabview()  # Update the images view
        else:  # if the split checkbox value is False, we try to de-split
            if (
                len(imagelist_items) > 1
            ):  # There should be multiple related splitted images
                self.remove_image(filepath)  # Remove the existing splitted images

                self.add_images(
                    {filepath: self.filepaths[filepath]}
                )  # Add back to original image

                self.update_images_tabview()  # update the images view

    def set_hyperstack(self, hyperstack_checkbox_value: bool):
        """Stack or unstack all loaded images into a single hyperstack.

        When True, combines all loaded images along a new dimension.
        When False, restores the original individual images.

        Parameters
        ----------
        hyperstack_checkbox_value : bool
            True to hyperstack, False to restore originals.

        Returns
        -------
        int or None
            Returns -1 if hyperstacking fails. Returns None on success.
        """
        if hyperstack_checkbox_value:  # We try to hyperstack
            try:
                # Gather arrays and metadatas from all files
                arrays = []
                metadatas = []

                for path, (array, metadata) in self.filepaths.items():
                    arrays.append(array)
                    metadatas.append(metadata)

                # Hyperstack those
                hyperstacked_array, hyperstacked_metadata = hyperstack_arrays(
                    arrays, metadatas
                )

                # Prepare file name and path for the hyperstacked image
                hyperstacked_filename = f"hyperstack_{self.images[0].filename}"
                filepath = self.images[0].filepath
                hyperstacked_filepath = str(
                    Path(filepath).with_name(hyperstacked_filename)
                )

                # Update images
                self.clear_images()  # Remove old images
                self.add_images(
                    {hyperstacked_filepath: (hyperstacked_array, hyperstacked_metadata)}
                )  # Add the hyperstacked image

                aprint("Images stacked")
            except Exception as e:
                aprint(e)
                return -1
        else:  # We try to de-hyperstack
            filepaths = self.filepaths  # Keep the original file list
            self.clear_filepaths()  # Clear all the files and images
            self.add_filepaths(filepaths)  # Add the original file list back

    def add_arrays(self, arrays_dict):
        """Add in-memory arrays (e.g. from napari layers) to the data model.

        Unlike :meth:`add_filepaths`, this does not read from disk.
        Each entry uses a ``napari://<name>`` synthetic path so the
        File(s) tab can display and manage it consistently.

        Parameters
        ----------
        arrays_dict : dict
            Mapping of ``{name: (ndarray, FileMetadata)}`` pairs.
        """
        new_images = {}
        for name, (array, metadata) in arrays_dict.items():
            synthetic_path = f'napari://{name}'
            if synthetic_path in self._filepaths:
                continue
            self._filepaths[synthetic_path] = (array, metadata)
            new_images[synthetic_path] = (array, metadata)

        if new_images:
            self.update_files_tabview()
            self.add_images(new_images)

    def update_files_tabview(self):
        """Trigger a refresh of the File(s) tab view."""
        self.parent.filestab_changed()

    def update_images_tabview(self):
        """Trigger a refresh of the Image(s), Dimensions, and Cropping tab views."""
        self.parent.imagestab_changed()
        self.update_dimensions_tabview()
        self.update_cropping_tabview()

    def update_dimensions_tabview(self):
        """Trigger a refresh of the Dimensions tab view."""
        self.parent.dimensionstab_changed()

    def update_cropping_tabview(self):
        """Trigger a refresh of the Training Crop and Denoising Crop tab views."""
        self.parent.croppingtabs_changed()

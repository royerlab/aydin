from copy import deepcopy
from os import listdir
from os.path import join, isfile, isdir
from pathlib import Path

from aydin.io import imread
from aydin.io.utils import split_image_channels, hyperstack_arrays
from aydin.util.log.log import lprint


class DataModel:
    """DataModel - manages our business logic regarding
    the images loaded into our GUI.

    Parameters
    ----------
    parent : Object
    """

    def __init__(self, parent=None):
        self.parent = parent
        self._filepaths = dict()
        self._images = []

    @property
    def filepaths(self):
        """filepaths is the dict which has file paths as keys
        and corresponding tuple of array and metadata as value.

        Returns
        -------
        self._filepaths : dict

        """
        return self._filepaths

    def clear_filepaths(self):
        """Clears the self._filepaths attribute and images of
        the model and triggers needed updates on files view.
        """
        self._filepaths = dict()
        self.update_files_tabview()
        self.clear_images()

    def add_filepaths(self, file_paths):
        """Adds filepaths to the model if they do not already
        exist in the current state of the model. If any file
        path is added, method also triggers addition of the
        corresponding images to the model too.

        Parameters
        ----------
        file_paths : List[str]

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
                else:
                    new_paths_added = True
                    self._filepaths[file_path] = (array, metadata)
                    new_images[file_path] = (array, metadata)

        if new_paths_added:
            self.update_files_tabview()
            self.add_images(new_images)

    def remove_filepaths(self, fpaths):
        """Removes given file paths and their corresponding
        images from the model.

        Parameters
        ----------
        fpaths : List[str]

        """
        for fpath in fpaths:
            del self._filepaths[fpath]
            self.remove_image(fpath)

    @property
    def images(self):
        """images is the list of a list with a particular
        format for each existing image in the model.

        Returns
        -------
        self._images : List[List[str, numpy.typing.ArrayLike, FileMetadata, bool, bool, str]]

        """
        return self._images

    def clear_images(self):
        """Clears the self._images attribute and triggers
        needed updates on images view.
        """
        self._images.clear()
        self.update_images_tabview()

    def add_images(self, new_images, train_on=True, denoise=True):
        """Adds images to the model. Method also triggers
        needed updates on images view.

        Parameters
        ----------
        new_images : dict
            Expects a particular dict format where key is
            the filepath and value is tuple of corresponding
            array and metadata.
        train_on : bool, optional
        denoise : bool, optional

        """
        for path, (array, metadata) in new_images.items():
            self._images.append(
                [Path(path).name, array, metadata, train_on, denoise, path]
            )

        self.update_images_tabview()

    def remove_image(self, image_filepath):
        """Removes all images from the model for the given
        file path.

        Parameters
        ----------
        image_filepath : str

        """
        indices2remove = []
        for idx, imagelist_item in enumerate(self._images):
            if (
                Path(image_filepath).parents[0] == Path(imagelist_item[5]).parents[0]
                and Path(image_filepath).name in imagelist_item[0]
            ):
                indices2remove.append(idx)

        for idx in reversed(indices2remove):
            self._images.pop(idx)

        self.update_images_tabview()

    @property
    def images_to_denoise(self):
        """Returns sublist of self.images with only elements
        of it that are marked to be denoised.

        Returns
        -------
        List[List[str, numpy.typing.ArrayLike, FileMetadata, bool, bool, str]]

        """
        return list(filter(lambda image: image[4], self.images))

    def set_image_to_denoise(self, filename, new_value):
        """Method to mark/unmark to denoise an image the with
        corresponding filename.

        Parameters
        ----------
        filename : str
        new_value : bool

        """
        for imagelist_item in self._images:
            if imagelist_item[0] == filename:
                imagelist_item[4] = new_value
                self.update_dimensions_tabview()
                self.update_cropping_tabview()

    def set_split_channels(self, filename, filepath, new_value: bool):
        """Method to split/de-split channels of an image with
        corresponding filename.

        Parameters
        ----------
        filename : str
        filepath : str
        new_value : bool

        Returns
        -------
        int
            returns -1 if the image doesn't have a channel axis.

        """
        # Fetch the self._images elements that are associated by their file names
        imagelist_items = [elem for elem in self._images if filename in elem[0]]

        if new_value:  # if the split checkbox value is True, we try to split

            # There should be only one related image as a non-splitted file
            if len(imagelist_items) != 1:
                return

            imagelist_item = imagelist_items[0]

            if imagelist_item[2].axes.find("C") == -1:
                lprint("Array has no channel axis detected")
                return -1

            self.remove_image(filepath)  # Remove the existing image from the model

            # Split the array and collect new arrays and metadatas
            splitted_arrays, metadatas = split_image_channels(
                imagelist_item[1], deepcopy(imagelist_item[2])
            )

            # Prepare new file names for the splitted images
            filenames = [f"channel_{_}_{filename}" for _ in range(len(splitted_arrays))]

            # Add the splitted images to the model
            new_images = {}
            for filename, array, metadata in zip(filenames, splitted_arrays, metadatas):
                new_images[filepath.replace(Path(filepath).name, filename)] = (
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
        """Method to hyperstack/de-hyperstack files in the model.

        Parameters
        ----------
        hyperstack_checkbox_value : bool

        Returns
        -------
        int
            returns -1 if hyperstacking fails for any reason.

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
                hyperstacked_filename = f"hyperstack_{self.images[0][0]}"
                filepath = self.images[0][5]
                hyperstacked_filepath = filepath.replace(
                    Path(filepath).name, hyperstacked_filename
                )

                # Update images
                self.clear_images()  # Remove old images
                self.add_images(
                    {hyperstacked_filepath: (hyperstacked_array, hyperstacked_metadata)}
                )  # Add the hyperstacked image

                lprint("Images stacked")
            except Exception as e:
                print(e)
                return -1
        else:  # We try to de-hyperstack
            filepaths = self.filepaths  # Keep the original file list
            self.clear_filepaths()  # Clear all the files and images
            self.add_filepaths(filepaths)  # Add the original file list back

    def update_files_tabview(self):
        """Method to trigger updating files tab view."""
        self.parent.filestab_changed()

    def update_images_tabview(self):
        """Method to trigger updating images tab view."""
        self.parent.imagestab_changed()
        self.update_dimensions_tabview()
        self.update_cropping_tabview()

    def update_dimensions_tabview(self):
        """Method to trigger updating dimensions tab view."""
        self.parent.dimensionstab_changed()

    def update_cropping_tabview(self):
        """Method to trigger updating cropping tab view."""
        self.parent.croppingtabs_changed()

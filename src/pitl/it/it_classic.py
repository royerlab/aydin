import math
import tempfile
from functools import reduce
from operator import mul
from random import shuffle

import numpy
import psutil

from pitl.it.it_base import ImageTranslatorBase
from pitl.regression.gbm import GBMRegressor
from pitl.util.combinatorics import closest_product
from pitl.util.nd import nd_split_slices, remove_margin_slice
from src.pitl.features.mcfocl import MultiscaleConvolutionalFeatures


class ImageTranslatorClassic(ImageTranslatorBase):
    """
        Portable Image Translation Learning (PITL)

        Using classic ML (feature generation + regression)

    """

    def __init__(self,
                 feature_generator=MultiscaleConvolutionalFeatures(),
                 regressor=GBMRegressor()
                 ):
        """

        :param feature_generator:
        :type feature_generator:
        :param regressor:
        :type regressor:
        """
        self.debug = True
        self.models = []

        self.feature_generator = feature_generator
        self.regressor = regressor

        self.receptive_field_radius = self.feature_generator.get_receptive_field_radius()

        self.self_supervised = None

    def _compute_features(self, image, exclude_center, batch_dims):
        """

        :param image:
        :type image:
        :param exclude_center:
        :type exclude_center:
        :return:
        :rtype:
        """
        if self.debug:
            print("Computing features ")

        self.feature_generator.exclude_center = exclude_center
        features = self.feature_generator.compute(image, batch_dims)
        x = features.reshape(-1, features.shape[-1])

        return x

    def _predict_from_features(self,
                               x,
                               input_image_shape):
        """
            internal function that predicts y from the features x
        :param x:
        :type x:
        :param input_image_shape:
        :type input_image_shape:
        :param clip:
        :type clip:
        :return:
        :rtype:
        """
        yp = self.regressor.predict(x)
        yp = numpy.clip(yp, 0, 1)
        yp = yp.astype(numpy.float32)
        inferred_image = yp.reshape(input_image_shape)
        return inferred_image

    def train(self,
              input_image,
              target_image,
              train_test_ratio=0.1,
              batch_dims=None,
              batch_size=None,
              batch_shuffle = False):
        """
            Train to translate a given input image to a given output image

        """

        self.regressor.reset()

        if self.debug:
            print(f"Training on image of dimension {str(input_image.shape)} .")

        self.self_supervised = input_image is target_image

        shape = input_image.shape

        # set default batch_dim value:
        if batch_dims is None:
            batch_dims = (False,) * len(input_image.shape)

        dataset_size_in_bytes = input_image.size * input_image.itemsize + 0 if self.self_supervised else (target_image.size * target_image.itemsize)
        dataset_size_for_features_in_bytes = self.feature_generator.get_needed_mem(input_image.size)

        if self.debug:
            print(f'Dataset is {(dataset_size_in_bytes/1E6)} MB.')
            print(f'Dataset is {(dataset_size_for_features_in_bytes/1E6)} MB when computing features (float type).')

        free_cpu_mem_in_bytes = psutil.virtual_memory().free
        free_feature_mem_in_bytes = self.feature_generator.get_free_mem()
        if self.debug:
            print(f'There is {int(free_cpu_mem_in_bytes/1E6)} MB of free CPU memory')
            print(f'There is {int(free_feature_mem_in_bytes/1E6)} MB of free feature gen memory (GPU)')

        # We specify how much more free memory we need compared to the size of the data we need to allocate:
        # This is to give us some room. In the case of feature generation, it seems that GPU mem fragmentation prevents
        # us from allocating really big chunks...
        cpu_loading_factor = 1.2
        feature_loading_factor = 10.0

        cpu_batch_size = free_cpu_mem_in_bytes // cpu_loading_factor
        feature_gen_batch_size = free_feature_mem_in_bytes // feature_loading_factor
        max_batch_size = min(cpu_batch_size, feature_gen_batch_size)

        if batch_size is None:
            effective_batch_size = max_batch_size
        else:
            effective_batch_size = min(batch_size, max_batch_size)

        if self.debug:
            print(f'Batch size is: {(effective_batch_size/1E6)} MB.')

        is_enough_memory = dataset_size_in_bytes <= effective_batch_size and dataset_size_for_features_in_bytes <= effective_batch_size

        if (batch_size is None) and is_enough_memory:
            if self.debug:
                print(f'There is enough memory -- we do full training (no batches, i.e. single batch).')
            # array = np.zeros(shape, dtype=np.float32)

            self.batch_strategy = None

            inferred_image = self._train(input_image, target_image, batch_dims, train_test_ratio, batch=False)
            return inferred_image

        else:
            if self.debug:
                print(f'There is not enough memory (CPU or GPU) -- we have to do batch training.')


            # Ok, now we can start iterating through the batches:
            # strategy is a list in which each integer is the number of chunks per dimension.

            self.batch_strategy = self._get_batching_strategy(batch_dims, dataset_size_in_bytes, effective_batch_size, shape)

            margins = self.get_margins(shape, self.batch_strategy)

            batch_slices = nd_split_slices(shape, self.batch_strategy, do_shuffle = batch_shuffle, margins=margins)

            for slice_tuple in batch_slices:
                input_image_batch  = input_image[slice_tuple]
                target_image_batch = target_image[slice_tuple]

                self._train(input_image_batch, target_image_batch, batch_dims, train_test_ratio, batch=True)

            return None

    def _get_batching_strategy(self, batch_dims, dataset_size_in_bytes, effective_batch_size, shape):
        # We will store the batch strategy as a list of integers representing the number of chunks per dimension:
        batch_strategy = None
        # This is the ideal number of batches so that we partition just enough:
        ideal_number_of_batches = max(1, int(math.ceil(dataset_size_in_bytes // effective_batch_size)))
        # This is the total number of batches that we would get if we were to use all batch dimensions:
        num_provided_batches = reduce(mul, [(dim_size if is_batch else 1) for dim_size, is_batch in zip(shape, batch_dims)], 1)
        # We can use that to already determine whether the provided batch dimensions are enough:
        if num_provided_batches >= ideal_number_of_batches:
            # In this case we can make use of these dimensions, but there might be too many batches if we don't chunk...

            # Batch dimensions -- other dimensions are set to 1:
            batch_dimensions_sizes = [(dim_size if is_batch else 1) for dim_size, is_batch in zip(shape, batch_dims)]

            # Considering the dimensions marked by the client of this method as 'batch dimensions',
            # if possible, what combination of such dimensions is best for creating batches?
            best_batch_dimensions = closest_product(batch_dimensions_sizes, ideal_number_of_batches, 1.0, 5.0)

            # At this point, the strategy is simply to use the provided batch dimensions:
            batch_strategy = batch_dimensions_sizes

            # But, it might not be possible to do so because of too large batch dimensions...
            if best_batch_dimensions is None:
                # We could ignore this case, but this might lead to very inefficient processing due to excessive batching.
                # Also, regressors perform worse (e.g. lGBM) when there are many batches.
                # Please note that the feature generation and regression will take into account the informastion
                # about w

                # In the following we take the heuristic to split the longest batch dimension.

                # First, we identify the largest batch dimension:
                index_of_largest_batch_dimension = batch_dimensions_sizes.index(max(batch_dimensions_sizes))

                # Then we determine the number of batches excluding that dimension:
                num_batches_without_dimension = num_provided_batches / batch_dimensions_sizes[index_of_largest_batch_dimension]

                # Then we can determine the optimal batching for that dimension:
                optimum = int(math.ceil(ideal_number_of_batches / num_batches_without_dimension))

                # the strategy is then:
                batch_strategy[index_of_largest_batch_dimension] = optimum


        else:
            # In this case we have too few batches provided, so we need to further split the dataset:

            # This is the amount of batching still required beyond what batching dimensions provide:
            extra_batching = int(math.ceil(ideal_number_of_batches / num_provided_batches))

            # Now the question is how to distribute this among the other dimensions (non-batch).

            # First, what is the number of non-batch dimensions?
            num_non_batch_dimensions = sum([(0 if is_batch else 1) for is_batch in batch_dims])

            # Then, we distribute batching fairly based on the length of each dimension:

            # This is the product of non-batch dimensions:
            product_non_batch_dim = numpy.prod([dim_size for dim_size, is_batch in zip(shape, batch_dims) if not is_batch])

            # A little of logarythmic magic to find the 'geometric' (versus arithmetic) proportionality:
            alpha = math.log2(extra_batching) / math.log2(product_non_batch_dim)

            # The strategy is then:
            batch_strategy = [(dim_size if is_batch else int(math.ceil(dim_size ** alpha))) for dim_size, is_batch in zip(shape, batch_dims)]
        if self.debug:
            print(f"Batching strategy is: {batch_strategy}")

        return batch_strategy

    def _train(self, input_image, target_image, batch_dims, train_test_ratio, batch=False):
        x = self._compute_features(input_image, self.self_supervised, batch_dims)
        y = target_image.reshape(-1)
        # if self.debug:
        #    assert numpy.may_share_memory(target_image, y)
        nb_features = x.shape[-1]
        nb_entries = y.shape[0]
        if self.debug:
            print("Number of entries: %d features: %d ." % (nb_entries, nb_features))
            print("Splitting train and test sets.")
        # creates random complementary indices for selecting train and test entries:
        test_size = int(train_test_ratio * nb_entries)
        train_indices = numpy.full(nb_entries, False)
        train_indices[test_size:] = True
        numpy.random.shuffle(train_indices)
        test_indices = numpy.logical_not(train_indices)
        # we allocate memory for the new arrays taking into account that we might need to use memory mapped files
        # in the splitting of train and test sets. The features are the heavy part, so that's what we map:
        x_train, y_train, x_test, y_test = (None,) * 4
        if isinstance(x, numpy.memmap):
            temp_file = tempfile.TemporaryFile()
            x_train = numpy.memmap(temp_file,
                                   dtype=numpy.float32,
                                   mode='w+',
                                   shape=((nb_entries - test_size), nb_features))
        else:
            x_train = numpy.zeros(((nb_entries - test_size), nb_features), dtype=numpy.float)
        y_train = numpy.zeros((nb_entries - test_size,), dtype=numpy.float)
        x_test = numpy.zeros((test_size, nb_features), dtype=numpy.float)
        y_test = numpy.zeros((test_size,), dtype=numpy.float)
        # train data
        numpy.copyto(x_train, x[train_indices])
        numpy.copyto(y_train, y[train_indices])
        # test data:
        numpy.copyto(x_test, x[test_indices])
        numpy.copyto(y_test, y[test_indices])
        if self.debug:
            print("Training...")
        if batch:
            self.regressor.fit_batch(x_train, y_train, x_valid=x_test, y_valid=y_test)
            return None
        else:
            self.regressor.fit(x_train, y_train, x_valid=x_test, y_valid=y_test)
            inferred_image = self._predict_from_features(x, input_image.shape)
            return inferred_image

    def translate(self, input_image, batch_dims=None):
        """
            Translates an input image into an output image according to the learned function
        :param input_image:
        :type input_image:
        :param clip:
        :type clip:
        :return:
        :rtype:
        """
        if self.debug:
            print("Predicting output image from input image of dimension %s ." % str(input_image.shape))

        if  self.batch_strategy is None:
            features = self._compute_features(input_image, self.self_supervised, batch_dims)

            inferred_image = self._predict_from_features(features,
                                                         input_image_shape=input_image.shape)
            return inferred_image

        else:
            #TODO: batch inference here...

            shape = input_image.shape

            inferred_image = numpy.zeros(shape, dtype=numpy.float32)

            margins = self.get_margins(shape, self.batch_strategy)

            if self.debug:
                print(f"Margins for batches: {margins} .")


            batch_slices_margins = nd_split_slices(shape, self.batch_strategy, margins=margins)
            batch_slices         = nd_split_slices(shape, self.batch_strategy)

            for slice_margin_tuple, slice_tuple in zip(batch_slices_margins,batch_slices):

                input_image_batch = input_image[slice_margin_tuple]

                features = self._compute_features(input_image_batch, self.self_supervised, batch_dims)

                inferred_image_batch = self._predict_from_features(features,
                                                             input_image_shape=input_image_batch.shape)

                remove_margin_slice_tuple = remove_margin_slice(shape, slice_margin_tuple, slice_tuple)

                inferred_image[slice_tuple] = inferred_image_batch[remove_margin_slice_tuple]


            return inferred_image


    def get_margins(self, shape, batch_strategy):
        # We compute the margin from the receptive field but limit it to 33% of the tile size:
        margins = tuple(min(self.receptive_field_radius, (dim // split)//3) for (dim, split) in zip(shape, batch_strategy))
        # We only need margins if we split a dimension:
        margins = tuple((0 if split == 1 else margin) for margin, split in zip(margins, batch_strategy))
        return margins

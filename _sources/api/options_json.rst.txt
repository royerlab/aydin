==============
Options JSON
==============

Options JSON files can be saved with help of Aydin Studio(GUI). They can be
passed to Aydin CLI to run on many more images with same options easily.

We basically have a python `dict` to store chosen options in the background.
When save options JSON functionality is triggered, we basically encode the
`dict` with help of `jsonpickle` package and we dump it into a new `.json`
file. To understand the format of options JSON, you can inspect the example
shared below:

.. code-block:: json

    "{
        \"feature_generator\": {
            \"class\": {
                \"py/type\": \"aydin.features.standard_features.StandardFeatureGenerator\"
            },
            \"kwargs\": {
                \"dct_max_freq\": 0.5,
                \"decimate_large_scale_features\": true,
                \"dtype\": {
                    \"py/reduce\": [
                        {\"py/type\": \"numpy.dtype\"},
                        {\"py/tuple\": [\"f4\", false, true]},
                        {\"py/tuple\": [3, \"<\", null, null, null, -1, -1, 0]}
                    ]
                },
                \"extend_large_scale_features\": false,
                \"include_corner_features\": false,
                \"include_dct_features\": false,
                \"include_fine_features\": true,
                \"include_line_features\": false,
                \"include_median_features\": false,
                \"include_random_conv_features\": false,
                \"include_scale_one\": false,
                \"include_spatial_features\": false,
                \"max_level\": 13,
                \"min_level\": 0,
                \"num_sinusoidal_features\": 0,
                \"scale_one_width\": 3,
                \"spatial_features_coarsening\": 2
            }
        },
        \"it\": {
            \"class\": {
                \"py/type\": \"aydin.it.fgr.ImageTranslatorFGR\"
            },
            \"kwargs\": {
                \"balance_training_data\": false,
                \"favour_bright_pixels\": false,
                \"max_voxels_for_training\": null,
                \"voxel_keep_ratio\": 1.0
            }
        },
        \"regressor\": {
            \"class\": {
                \"py/type\": \"aydin.regression.cb.CBRegressor\"
            },
            \"kwargs\": {
                \"compute_load\": 0.95,
                \"gpu\": true,
                \"gpu_devices\": null,
                \"learning_rate\": 0.01,
                \"loss\": \"l1\",
                \"max_bin\": null,
                \"max_num_estimators\": 2048,
                \"min_num_estimators\": 512,
                \"num_leaves\": 512,
                \"patience\": 32
            }
        },
        \"variant\": \"Noise2SelfFGR-cb\"
    }"





Feature Generator
------------------

Aydin use a set of well-engineered feature generators to implement image
translators internally. Aydin also provides a public API on feature generators
to enable developers who might want to use same feature:


StandardFeatureGenerator
    .. currentmodule:: aydin.features.standard_features

    .. autosummary::
        StandardFeatureGenerator


ExtensibleFeatureGenerator
    .. currentmodule:: aydin.features.extensible_features

    .. autosummary::
        ExtensibleFeatureGenerator
        ExtensibleFeatureGenerator.add_feature_group
        ExtensibleFeatureGenerator.compute
        ExtensibleFeatureGenerator.create_feature_array
        ExtensibleFeatureGenerator.clear_features
        ExtensibleFeatureGenerator.get_num_features
        ExtensibleFeatureGenerator.get_receptive_field_radius
        ExtensibleFeatureGenerator.load
        ExtensibleFeatureGenerator.save


FeatureGeneratorBase
    .. currentmodule:: aydin.features.base

    .. autosummary::
        FeatureGeneratorBase
        FeatureGeneratorBase.compute
        FeatureGeneratorBase.create_feature_array
        FeatureGeneratorBase.get_receptive_field_radius
        FeatureGeneratorBase.load
        FeatureGeneratorBase.save


StandardFeatureGenerator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aydin.features.standard_features

.. autoclass:: StandardFeatureGenerator
    :members:
    :inherited-members:


ExtensibleFeatureGenerator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aydin.features.extensible_features

.. autoclass:: ExtensibleFeatureGenerator
    :members:
    :inherited-members:


FeatureGeneratorBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: aydin.features.base

.. autoclass:: FeatureGeneratorBase
    :members:
    :inherited-members:

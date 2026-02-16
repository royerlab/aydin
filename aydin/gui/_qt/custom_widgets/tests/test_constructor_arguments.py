"""Tests for ConstructorArgumentsWidget helper methods."""

from aydin.gui._qt.custom_widgets.constructor_arguments import (
    ConstructorArgumentsWidget,
)


class _ClassWithNoneInDoc:
    """Example class for testing."""

    def __init__(self, learning_rate=None, verbose=True):
        """Initialize.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate. If the value None is given, the rate is
            determined automatically.
        verbose : bool, optional
            Whether to print progress.
        """
        pass


class _ClassWithHiddenParam:
    """Example class for testing hidden parameters."""

    def __init__(self, alpha=0.5, _internal=None):
        """Initialize.

        Parameters
        ----------
        alpha : float
            Regularization parameter.
        _internal : object
            (hidden) Internal implementation detail.
        """
        pass


class _ClassWithAdvancedParam:
    """Example class for testing advanced parameters."""

    def __init__(self, sigma=1.0, epsilon=1e-8):
        """Initialize.

        Parameters
        ----------
        sigma : float
            Standard deviation.
        epsilon : float
            (advanced) Small constant for numerical stability.
        """
        pass


class _ClassWithNoDocstring:
    def __init__(self):
        pass


class _ClassWithEmptyDescription:
    """Example class with a parameter that has no description."""

    def __init__(self, x=1):
        """Initialize.

        Parameters
        ----------
        x : int
        """
        pass


def _function_with_doc(x=1, y=2):
    """A function.

    Parameters
    ----------
    x : int
        The x value.
    y : int
        The y value.
    """
    pass


def test_parse_param_descriptions_basic():
    """Test basic parsing returns correct parameter descriptions."""
    result = ConstructorArgumentsWidget._parse_param_descriptions(_ClassWithNoneInDoc)
    assert 'learning_rate' in result
    assert 'verbose' in result
    assert (
        'None' in result['learning_rate']
    ), "The word 'None' should be preserved in descriptions"


def test_parse_param_descriptions_none_preserved():
    """Test that 'None' in descriptions is NOT replaced with 'auto'."""
    result = ConstructorArgumentsWidget._parse_param_descriptions(_ClassWithNoneInDoc)
    desc = result['learning_rate']
    # The description should contain "None" as-is, not "'auto'"
    assert 'None' in desc
    assert "'auto'" not in desc


def test_parse_param_descriptions_hidden_param_present():
    """Test that hidden parameters are still in the parsed dict (filtering happens elsewhere)."""
    result = ConstructorArgumentsWidget._parse_param_descriptions(_ClassWithHiddenParam)
    assert 'alpha' in result
    assert '_internal' in result
    assert '(hidden)' in result['_internal']


def test_parse_param_descriptions_advanced_param():
    """Test that advanced parameters are parsed correctly."""
    result = ConstructorArgumentsWidget._parse_param_descriptions(
        _ClassWithAdvancedParam
    )
    assert 'sigma' in result
    assert 'epsilon' in result
    assert '(advanced)' in result['epsilon']


def test_parse_param_descriptions_none_class():
    """Test graceful handling of None reference class."""
    result = ConstructorArgumentsWidget._parse_param_descriptions(None)
    assert result == {}


def test_parse_param_descriptions_no_docstring():
    """Test graceful handling of class with no docstring."""
    result = ConstructorArgumentsWidget._parse_param_descriptions(_ClassWithNoDocstring)
    assert result == {}


def test_parse_param_descriptions_function():
    """Test parsing works with a function (not a class)."""
    result = ConstructorArgumentsWidget._parse_param_descriptions(_function_with_doc)
    assert 'x' in result
    assert 'y' in result


def test_parse_param_descriptions_correct_mapping():
    """Test that descriptions map to the correct parameter names."""
    result = ConstructorArgumentsWidget._parse_param_descriptions(_ClassWithNoneInDoc)
    assert (
        'learning rate' in result['learning_rate'].lower()
        or 'rate' in result['learning_rate'].lower()
    )
    assert (
        'progress' in result['verbose'].lower() or 'print' in result['verbose'].lower()
    )


def test_parse_param_descriptions_none_description_becomes_empty_string():
    """Test that a param with no description text returns '' not None."""
    result = ConstructorArgumentsWidget._parse_param_descriptions(
        _ClassWithEmptyDescription
    )
    assert 'x' in result
    assert result['x'] == "", f"Expected empty string, got {result['x']!r}"

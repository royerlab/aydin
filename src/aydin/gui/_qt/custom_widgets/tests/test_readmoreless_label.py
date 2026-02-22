"""Tests for QReadMoreLessLabel widget."""

import pytest
from qtpy.QtWidgets import QWidget

from aydin.gui._qt.custom_widgets.readmoreless_label import QReadMoreLessLabel

pytestmark = pytest.mark.gui


class TestSplitOnly:
    """Tests for text with <split> but no <moreless>."""

    def test_left_right_columns(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        label = QReadMoreLessLabel(parent, "Left content<split>Right content")
        assert "Left content" in label.explanation_text_left.text()
        assert "Right content" in label.explanation_text_right.text()

    def test_no_readmore_state(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        label = QReadMoreLessLabel(parent, "Left<split>Right")
        assert label.readmore is False
        assert label.readmore_text is None


class TestMoreLess:
    """Tests for text with <moreless> and <split>."""

    SAMPLE_TEXT = (
        "Short summary<moreless> Extended left content<split>Extended right content"
    )

    def test_initial_collapsed_state(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        label = QReadMoreLessLabel(parent, self.SAMPLE_TEXT)
        assert label.readmore is False
        assert "Read more..." in label.explanation_text_left.text()
        assert label.explanation_text_right.text() == ""

    def test_toggle_expands(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        label = QReadMoreLessLabel(parent, self.SAMPLE_TEXT)
        label.state_toggle()
        assert label.readmore is True
        assert "Read less..." in label.explanation_text_right.text()
        # Left column should show the full readmore_left text
        assert "Extended left content" in label.explanation_text_left.text()

    def test_toggle_collapses_again(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        label = QReadMoreLessLabel(parent, self.SAMPLE_TEXT)
        label.state_toggle()  # expand
        label.state_toggle()  # collapse
        assert label.readmore is False
        assert "Read more..." in label.explanation_text_left.text()
        assert label.explanation_text_right.text() == ""


class TestEmptyRightSide:
    """Tests for text with empty right side after <split>."""

    def test_empty_right_hides_separator(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        text = "Summary<moreless> Left content<split>  "
        label = QReadMoreLessLabel(parent, text)
        # When readmore_right is whitespace-only, "Read more..." is not shown
        assert "Read more..." not in label.explanation_text_left.text()


class TestRealDocstring:
    """Test with actual DimensionsTab docstring to verify no crash."""

    def test_dimensions_tab_docstring(self, qtbot):
        from aydin.gui.tabs.qt.dimensions import DimensionsTab

        parent = QWidget()
        qtbot.addWidget(parent)
        doc = DimensionsTab.__doc__
        # Should not raise
        label = QReadMoreLessLabel(parent, doc)
        assert label is not None
        # <moreless> tag should have been parsed
        assert label.readmore_text is not None

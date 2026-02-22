"""Tests for OutputWrapper."""

import sys

import pytest
from qtpy.QtWidgets import QWidget

from aydin.gui._qt.output_wrapper import OutputWrapper

pytestmark = pytest.mark.gui


class TestStdoutRedirect:
    """Tests for stdout redirection."""

    def test_stdout_is_replaced(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        original = sys.stdout
        try:
            wrapper = OutputWrapper(parent, stdout=True)
            assert sys.stdout is wrapper
            assert wrapper._stream is original
        finally:
            sys.stdout = original

    def test_output_written_signal(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        original = sys.stdout
        try:
            wrapper = OutputWrapper(parent, stdout=True)
            with qtbot.waitSignal(wrapper.outputWritten, timeout=1000) as sig:
                wrapper.write("hello")
            assert sig.args == ["hello"]
        finally:
            sys.stdout = original

    def test_del_restores_stdout(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        original = sys.stdout
        try:
            wrapper = OutputWrapper(parent, stdout=True)
            assert sys.stdout is wrapper
            wrapper.__del__()
            assert sys.stdout is original
        finally:
            sys.stdout = original


class TestStderrRedirect:
    """Tests for stderr redirection."""

    def test_stderr_is_replaced(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        original = sys.stderr
        try:
            wrapper = OutputWrapper(parent, stdout=False)
            assert sys.stderr is wrapper
            assert wrapper._stream is original
        finally:
            sys.stderr = original

    def test_stderr_output_written_signal(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        original = sys.stderr
        try:
            wrapper = OutputWrapper(parent, stdout=False)
            with qtbot.waitSignal(wrapper.outputWritten, timeout=1000) as sig:
                wrapper.write("error msg")
            assert sig.args == ["error msg"]
        finally:
            sys.stderr = original

    def test_del_restores_stderr(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        original = sys.stderr
        try:
            wrapper = OutputWrapper(parent, stdout=False)
            assert sys.stderr is wrapper
            wrapper.__del__()
            assert sys.stderr is original
        finally:
            sys.stderr = original


class TestDelegation:
    """Tests for attribute delegation to the original stream."""

    def test_flush_delegates(self, qtbot):
        parent = QWidget()
        qtbot.addWidget(parent)
        original = sys.stdout
        try:
            wrapper = OutputWrapper(parent, stdout=True)
            # flush should not raise — it delegates to the original stream
            wrapper.flush()
        finally:
            sys.stdout = original

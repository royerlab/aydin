"""Tests for Worker and WorkerSignals."""

import pytest

from aydin.gui._qt.job_runners.worker import Worker

pytestmark = pytest.mark.gui


class TestWorkerSuccess:
    """Tests for successful worker execution."""

    def test_result_signal_emitted(self, qtbot):
        def task(progress_callback):
            return 42

        worker = Worker(task)
        results = []
        worker.signals.result.connect(results.append)

        with qtbot.waitSignal(worker.signals.finished, timeout=5000):
            worker.run()

        assert results == [42]

    def test_finished_signal_emitted(self, qtbot):
        def task(progress_callback):
            return "done"

        worker = Worker(task)

        with qtbot.waitSignal(worker.signals.finished, timeout=5000):
            worker.run()


class TestWorkerError:
    """Tests for worker error handling."""

    def test_error_signal_emitted(self, qtbot):
        def failing_task(progress_callback):
            raise ValueError("test error")

        worker = Worker(failing_task)
        errors = []
        worker.signals.error.connect(errors.append)

        with qtbot.waitSignal(worker.signals.finished, timeout=5000):
            worker.run()

        assert len(errors) == 1
        exctype, value, tb_str = errors[0]
        assert exctype is ValueError
        assert "test error" in str(value)

    def test_finished_emitted_on_error(self, qtbot):
        def failing_task(progress_callback):
            raise RuntimeError("boom")

        worker = Worker(failing_task)

        with qtbot.waitSignal(worker.signals.finished, timeout=5000):
            worker.run()


class TestWorkerProgress:
    """Tests for progress callback."""

    def test_progress_signal(self, qtbot):
        def task_with_progress(progress_callback):
            progress_callback.emit("50%")
            return "ok"

        worker = Worker(task_with_progress)
        progress_msgs = []
        worker.signals.progress.connect(progress_msgs.append)

        with qtbot.waitSignal(worker.signals.finished, timeout=5000):
            worker.run()

        assert "50%" in progress_msgs

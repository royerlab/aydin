"""Utility functions for iterating over QTreeWidget items."""

from qtpy.QtWidgets import QTreeWidgetItemIterator


def iter_tree_widget(root):
    """Iterate over all items in a QTreeWidget.

    Parameters
    ----------
    root : QTreeWidgetItem
        The root item (typically from ``tree.invisibleRootItem()``).

    Yields
    ------
    QTreeWidgetItem
        Each item in the tree in depth-first order.
    """
    iterator = QTreeWidgetItemIterator(root)
    while True:
        item = iterator.value()
        if item is not None:
            yield item
            iterator += 1
        else:
            break

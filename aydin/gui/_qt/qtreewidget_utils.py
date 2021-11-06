from qtpy.QtWidgets import QTreeWidgetItemIterator


def iter_tree_widget(root):
    iterator = QTreeWidgetItemIterator(root)
    while True:
        item = iterator.value()
        if item is not None:
            yield item
            iterator += 1
        else:
            break

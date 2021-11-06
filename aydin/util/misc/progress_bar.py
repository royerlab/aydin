from aydin.util.log.log import lprint


class ProgressBar:
    """Progress Bar

    Parameters
    ----------
    total : int

    """

    def __init__(self, total=100):
        self.total = total

    def emit(self, val):
        """Member method to emit

        Parameters
        ----------
        val : int

        """
        lprint("ProgressBar: ", val)
        if val > self.total:
            lprint("ProgressBar updated with a number bigger than it is maximum!")

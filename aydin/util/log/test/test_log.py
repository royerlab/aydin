import pytest

from aydin.util.log.log import lprint, lsection, Log


@pytest.mark.heavy
def test_log():

    # This is required for this test to pass!
    Log.override_test_exclusion = True

    lprint('Test')

    with lsection('a section'):
        lprint('a line')
        lprint('another line')
        lprint('we are done')

        with lsection('a subsection'):
            lprint('another line')
            lprint('we are done')

            with lsection('a subsection'):
                lprint('another line')
                lprint('we are done')

                assert Log.depth == 3

                with lsection('a subsection'):
                    lprint('another line')
                    lprint('we are done')

                    with lsection('a subsection'):
                        lprint('another line')
                        lprint('we are done')

                        assert Log.depth == 5

                        with lsection('a subsection'):
                            lprint('another line')
                            lprint('we are done')

                            with lsection('a subsection'):
                                lprint('another line')
                                lprint('we are done')

                                assert Log.depth == 7

                        with lsection('a subsection'):
                            lprint('another line')
                            lprint('we are done')

                    with lsection('a subsection'):
                        lprint('another line')
                        lprint('we are done')

                with lsection('a subsection'):
                    lprint('another line')
                    lprint('we are done')

    lprint('test is finished...')

    assert Log.depth == 0

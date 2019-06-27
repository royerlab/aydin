from pitl.io.download_examples import download_from_gdrive, examples, download_all_examples


def test_download():

    datadir = '../../../../data/examples'

    print(download_from_gdrive(*examples.generic_mandrill, datadir))


def test_all_download():

    print(examples.__dict__)

    download_all_examples()
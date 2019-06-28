from pitl.io.examples import download_from_gdrive, example_datasets, download_all_examples


def test_download():

    datadir = '../../../../data/examples'

    print(download_from_gdrive(*example_datasets.generic_mandrill, datadir))


def test_all_download():

    print(example_datasets.__dict__)

    download_all_examples()
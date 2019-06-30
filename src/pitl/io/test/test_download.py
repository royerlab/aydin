from pitl.io.datasets import download_from_gdrive, examples_single, download_all_examples, datasets_folder


def test_download():

    print(download_from_gdrive(*examples_single.generic_mandrill, datasets_folder))


def test_all_download():

    print(examples_single.__dict__)

    download_all_examples()
from pitl.io.datasets import download_from_gdrive, examples_single, download_all_examples, datasets_folder


def test_examples_single():
    for dataset in examples_single:
        print(dataset)


def test_download():
    print(download_from_gdrive(*examples_single.generic_mandrill.value, datasets_folder))


def test_all_download():

    download_all_examples()

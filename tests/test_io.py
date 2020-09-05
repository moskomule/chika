from chika.chika import is_supported_filetype


def test_is_supported_filetype():
    assert is_supported_filetype("test.yaml")
    assert is_supported_filetype("test.yml")
    assert is_supported_filetype("test.json")
    assert not is_supported_filetype("test.jpeg")
    assert not is_supported_filetype("test.png")

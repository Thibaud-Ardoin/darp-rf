from oru.posix import *
from pathlib import Path

def test_outputfile_stdin():
    with open_default_stdout("-") as f:
        f.write("hello\n")

    assert not Path('-').exists()


def test_input_outputfile_file():
    scrapfile = Path(__file__).parent / "scrap.txt"
    try:
        with open_default_stdout(scrapfile) as f:
            f.write("hello\n")

        assert scrapfile.exists(), "file not created"

        with open_default_stdin(scrapfile) as f:
            txt = f.read().strip()
        assert txt == "hello"

        with open_default_stdin(scrapfile, mode='rb') as f:
            txt = f.read()

        assert txt != "hello\n"
        assert txt == b"hello\n"
    except:
        raise
    finally:
        if scrapfile.exists():
            scrapfile.unlink()

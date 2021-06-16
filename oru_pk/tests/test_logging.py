from oru.logging import *
from pathlib import Path
import textwrap

def test_table_min_col_width():
    table = TablePrinter(["egg", "bacon", "minisoda"], min_col_width=7, sep="|", print_header=True)
    line = table.format_line(1, 2, 3)
    assert line == "      1|      2|       3"

def test_table_justify():
    table = TablePrinter(["1234", "12345", "1234"], justify="^", sep="|", print_header=True)
    line = table.format_line(1, 1, 11)
    assert line == " 1  |  1  | 11 "

def test_csv_logger_output():
    scrapfile = "scrap.csv"
    try:
        log = CSVLog(scrapfile,mode='w', index='epoch')
        log(epoch=0, foo='yes', bar=False, loss=0.9)
        log(epoch=1, foo='yes', bar=True, loss=0.8)
        log(foo='yes', loss=0.7, bar=False, epoch=2)
        log(epoch=3, foo='no', bar=False, loss=0.1)
        log.close()

        ref_contents = textwrap.dedent(
        """
        epoch,foo,bar,loss
        0,yes,False,0.9
        1,yes,True,0.8
        2,yes,False,0.7
        3,no,False,0.1
        """).lstrip('\n')

        with open(scrapfile) as f:
            contents = f.read()

        assert contents == ref_contents
    except:
        raise
    finally:
        os.remove(scrapfile)


def test_file_modes():
    scrapfile = Path("scrap.csv")
    scrapfile.touch(exist_ok=True)
    try:
        CSVLog(scrapfile)
    except FileExistsError:
        pass
    else:
        raise AssertionError("should have raised an error")
    finally:
        scrapfile.unlink()

    for mode in ("r", "r", "w+"):
        try:
            CSVLog(scrapfile, mode=mode)
        except ValueError:
            pass
        else:
            raise AssertionError("should have raised an error")


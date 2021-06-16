import svgpathtools
import bs4
import cmath
from . import posix

from typing import Union

def svg_align_edge_labels(src_svg : str,
                          dest_svg : Union[str, None] = None,
                          y_shift_px : float = 10):

    with posix.open_default_stdin(src_svg, mode='r') as f:
        xml = f.read()

    s = bs4.BeautifulSoup(xml, "xml")

    for edgetag in s.find_all(attrs={"class" : "edge"}):
        text = edgetag.find("text", recursive=False)
        if text is None:
            continue
        path = edgetag.find('path', recursive=False).attrs['d']
        if path is None:
            continue
        path = svgpathtools.parse_path(path)
        orientation = cmath.phase(path.unit_tangent(0.5))*180/cmath.pi
        center = path.point(0.5)
        x,y = center.real, center.imag
        y -= y_shift_px
        text.attrs["transform"] = f"rotate({orientation},{x},{y})"
        text.attrs['x'] = str(x)
        text.attrs['y'] = str(y)

    if dest_svg is None:
        dest_svg = src_svg

    with posix.open_default_stdout(dest_svg, mode="w") as f:
        f.write(str(s))

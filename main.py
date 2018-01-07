"""Main primipy module."""

from __future__ import print_function
from __future__ import division

import argparse
import copy
import os
import random

from PIL import Image
from PIL import ImageChops
from PIL import ImageDraw
from PIL import ImageStat
from PIL.PngImagePlugin import PngInfo

import svgwrite


def error(im1, im2):
    """Calculate the root-mean difference between two images."""
    im_i = ImageChops.difference(im1, im2)
    return sum(ImageStat.Stat(im_i).rms)


def clamp(low, x, up):
    """Clamp x between low and up."""
    return min(max(low, x), up)


def resize_to(im, dim):
    """Resize input image maintaining its aspect ratio.

    :param im: source image
    :type im: PIL.Image
    :param dim: maximum dimension to resize to
    :type dim: int
    :return: resized image
    :rtype: PIL.Image
    """
    im_ar = im.size[0] / im.size[1]
    if im.size[0] > im.size[1]:
        im_w = 256
        im_h = int(256 / im_ar)
    else:
        im_w = int(256 * im_ar)
        im_h = 256

    return im.resize((im_w, im_h))


class Shape(object):
    """A generic shape, abstract class."""

    def __init__(self, pts, col):
        """Shape constructor.

        :param src: source image
        :type src: PIL.Image
        """
        self.pts = pts
        self.col = col

    def __str__(self):
        """String-like representation for Shape."""
        return str((self.__class__.__name__, self.pts, self.col))


class Triangle(Shape):
    """Triangle shape."""

    def center(self):
        """Calculate shape center.

        :return: shape center
        :rtype: tuple
        """
        p1 = self.pts[0]
        p2 = self.pts[1]
        p3 = self.pts[2]
        mp = ((p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3)
        return mp

    def draw_pillow(self, imp, col):
        """Draw triangle on Pillow backend.

        :param imp: Pillow image proxy
        :type imp: ImageDraw.Draw
        :param col: shape color RGBA tuple
        :type col: tuple
        """
        imp.polygon(self.pts, col)

    def draw_svg(self, dwg, col):
        """Draw triangle on SVG backend.

        :param dwg: SVG drawing context
        :type dwg: svgwrite.Drawing
        :param col: shape color RGBA tuple
        :type col: tuple
        """
        col_svg = svgwrite.rgb(r=col[0], g=col[1], b=col[2], mode="RGB")
        col_alpha = str(col[3] / 255.0)
        polygon = dwg.polygon(points=self.pts, fill=col_svg, fill_opacity=col_alpha, clip_path="url(#c)")
        dwg.add(polygon)


class Rectangle(Shape):
    """Rectangle shape."""

    def center(self):
        """Calculate shape center.

        :return: shape center
        :rtype: tuple
        """
        p1 = self.pts[0]
        p2 = self.pts[1]
        mp = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        return mp

    def draw_pillow(self, imp, col):
        """Draw rectangle on Pillow backend.

        :param imp: Pillow image proxy
        :type imp: ImageDraw.Draw
        :param col: shape color RGBA tuple
        :type col: tuple
        """
        imp.rectangle(self.pts, col)

    def draw_svg(self, dwg, col):
        """Draw rectangle on SVG backend.

        :param dwg: SVG drawing context
        :type dwg: svgwrite.Drawing
        :param col: shape color RGBA tuple
        :type col: tuple
        """
        p1, p2 = self.pts
        col_svg = svgwrite.rgb(r=col[0], g=col[1], b=col[2], mode="RGB")
        col_alpha = str(col[3] / 255.0)
        rect = dwg.rect(insert=p1, size=(p2[0] - p1[0], p2[1] - p1[1]),
                        fill=col_svg, fill_opacity=col_alpha, clip_path="url(#c)")
        dwg.add(rect)


class State(object):
    """State object."""

    def __init__(self, src, dst, rects=None):
        """State constructor.

        State instances are immutable and represent the state of the picture.

        :param src: source image
        :type src: PIL.Image
        :param dst: destination image
        :type dst: PIL.Image
        :param rects: list of shapes rendered in destination image
        :type rects: list(tuple)
        """
        self.src = src
        self.dst = dst
        self.imp = ImageDraw.Draw(dst, "RGBA")

        if rects is None:
            self.rects = []
            # The very first shape is a full-screen one with the most common
            # color in the image

            colors = {}
            for y in xrange(src.size[1]):
                for x in xrange(src.size[0]):
                    col = src.getpixel((x, y))
                    colors.setdefault(col, 0)
                    colors[col] += 1

            avg_col = max(colors, key=lambda key: colors[key])
            avg_col = (avg_col[0], avg_col[1], avg_col[2], 255)
            r = Rectangle(pts=((0, 0), (self.src.size[0] - 1, self.src.size[1] - 1)), col=avg_col)
            self.rects.append(r)
        else:
            # Assume client provides a copy
            self.rects = rects

        self.render()

    def randrect(self):
        """Generate a random rectangle."""
        maxw = self.dst.size[0]
        maxh = self.dst.size[1]

        x, y = random.randint(0, maxw), random.randint(0, maxh)
        w, h = random.randint(0, maxw), random.randint(0, maxh)

        w /= 2
        h /= 2

        p1 = (x - w / 2, y - h / 2)
        p2 = (x + w / 2, y + h / 2)
        return Rectangle(pts=(p1, p2), col=None)

    def randtri(self):
        """Generate a random triangle."""
        maxw = self.dst.size[0]
        maxh = self.dst.size[1]
        mw = maxw / 4
        mh = maxh / 4

        cx, cy = random.randint(0, maxw), random.randint(0, maxh)

        x1, y1 = cx + random.randint(-mw, mw), cy + random.randint(-mh, mh)
        x2, y2 = cx + random.randint(-mw, mw), cy + random.randint(-mh, mh)
        x3, y3 = cx + random.randint(-mw, mw), cy + random.randint(-mh, mh)

        return Triangle(pts=((x1, y1), (x2, y2), (x3, y3)), col=None)

    def slopedtri(self):
        """Generate a sloped triangle shape."""
        maxw = self.dst.size[0]
        maxh = self.dst.size[1]

        x, y = random.randint(0, maxw), random.randint(0, maxh)
        w, h = random.randint(0, maxw), random.randint(0, maxh)

        p1 = (x - w / 2, y - h / 2)
        p2 = (x + w / 2, y + h / 2)
        p3 = (p1[0], p2[1])

        return Triangle(pts=(p1, p2, p3), col=None)

    def improve(self):
        """Add a random shape to shape list to create a new State.

        :return: a new State instance
        :rtype: State
        """
        r = self.randtri()

        # Copies the destination image, including its last rendering
        ndst = copy.copy(self.dst)

        nrects = copy.copy(self.rects)
        nrects.append(r)

        return State(src=self.src, dst=ndst, rects=nrects)

    def error(self):
        """Compute current error against source reference.

        :return: error metric
        :rtype: int
        """
        return error(self.src, self.dst)

    def render(self):
        """Render the current state of the shapes."""
        for shape in [self.rects[-1]]:
            c = shape.col
            if c is None:
                mp = shape.center()
                mp = (clamp(0, mp[0], self.src.size[0] - 1), clamp(0, mp[1], self.src.size[1] - 1))

                pix = self.src.getpixel(mp)

                col = (pix[0], pix[1], pix[2], 0x77)
            else:
                col = c

            shape.draw_pillow(self.imp, col)

    def dump_to_svg(self, filename):
        """Render the current state of the shapes using SVG."""
        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=self.dst.size[0], height=self.dst.size[1])

        # Add a clipping path to make sure the drawing is limited to the
        # destination image canvas.
        clip_path = dwg.defs.add(dwg.clipPath(id="c"))
        clip_path.add(dwg.rect(insert=(0, 0), size=(self.dst.size[0], self.dst.size[1])))

        for shape in self.rects:
            c = shape.col
            if c is None:
                mp = shape.center()
                mp = (clamp(0, mp[0], self.src.size[0] - 1), clamp(0, mp[1], self.src.size[1] - 1))

                pix = self.src.getpixel(mp)

                col = (pix[0], pix[1], pix[2], 0x77)
            else:
                col = c

            shape.draw_svg(dwg, col)

        dwg.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="print information to stdout", action="store_true")
    parser.add_argument("-i", dest="input", help="input image", required=True)
    parser.add_argument("-o", dest="output", help="output image", required=True)
    parser.add_argument("-n", dest="nshapes", type=int, help="number of shapes", required=True)
    parser.add_argument("-iters", dest="niters", type=int, help="number of iterations", default=100)

    args = parser.parse_args()

    im = Image.open(args.input).convert("RGB")
    im = resize_to(im, 256)

    im2 = Image.new("RGB", (im.width, im.height))

    state = State(im, im2)

    best_overall_so_far = state
    best_overall_error = state.error()
    # Number of shapes in the image
    for a in range(args.nshapes):

        best_so_far = None
        best_error = None

        # New polygons to try
        for b in range(args.niters):

            ns = best_overall_so_far.improve()

            if best_so_far is None:
                best_so_far = ns
                best_error = ns.error()
                continue

            err = ns.error()
            if err < best_error and err < best_overall_error:
                best_so_far = ns
                best_error = err

        if best_error < best_overall_error:
            best_overall_error = best_error
            best_overall_so_far = best_so_far
            if args.verbose:
                print("Iter", a, "Error improved to", best_error)

        best_overall_so_far.dst.save(args.output)

    # Save final PNG, with generator information
    png_info = PngInfo()
    png_info.add_text("generator", "primipy")
    png_info.add_text("nshapes", str(args.nshapes))
    png_info.add_text("niters", str(args.niters))
    best_overall_so_far.dst.save(args.output, pnginfo=png_info)

    # Save final SVG
    svg_filename = ".".join([os.path.splitext(args.output)[0], "svg"])
    best_overall_so_far.dump_to_svg(svg_filename)

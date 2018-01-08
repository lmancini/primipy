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


def dominant_color(im):
    """Get dominant color in image.

    :param im: source image
    :type im: PIL.Image
    :return: dominant color (RGBA)
    :rtype: tuple
    """
    imp = im.convert("P", palette=Image.ADAPTIVE, colors=2)
    imp.putalpha(255)
    colors = imp.getcolors(2)
    __, dominant_color = colors[0]
    return dominant_color


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

    def mutate(self):
        """Mutate this Triangle.

        :return: mutated Triangle
        :rtype: Triangle
        """
        m_idx = random.randint(0, 2)
        p = self.pts[m_idx]
        np = (p[0] + random.randint(-16, 16), p[1] + random.randint(-16, 16))

        n_pts = [self.pts[i] if i != m_idx else np for i in range(3)]
        return Triangle(pts=n_pts, col=self.col)

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

    def mutate(self):
        """Mutate this Rectangle.

        :return: mutated Rectangle
        :rtype: Rectangle
        """
        m_idx = random.randint(0, 1)
        p = self.pts[m_idx]
        np = (p[0] + random.randint(-16, 16), p[1] + random.randint(-16, 16))

        n_pts = [self.pts[i] if i != m_idx else np for i in range(2)]
        return Rectangle(pts=n_pts, col=self.col)

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

    def __init__(self, src, dst, rects=None, _mutation=None):
        """State constructor.

        State instances are immutable and represent the state of the picture.

        :param src: source image
        :type src: PIL.Image
        :param dst: destination image
        :type dst: PIL.Image
        :param rects: list of shapes rendered in destination image
        :type rects: list(tuple)
        :param _mutation: None if this state is not a mutation
                          False if this state just ceased being one
                          True if this state is a mutation
        :type _mutation: bool | None
        """
        self.src = src
        self.dst = dst
        self.imp = ImageDraw.Draw(dst, "RGBA")

        self._mutation = _mutation

        if rects is None:
            self.rects = []
            # The very first shape is a full-screen one with the dominant
            # color in the image
            dom_col = dominant_color(src)

            r = Rectangle(pts=((0, 0), (self.src.size[0] - 1, self.src.size[1] - 1)), col=dom_col)
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

        x1, y1 = random.randint(0, maxw), random.randint(0, maxh)

        x2 = x1 + random.randint(0, 31) - 15
        y2 = y1 + random.randint(0, 31) - 15
        x3 = x1 + random.randint(0, 31) - 15
        y3 = y1 + random.randint(0, 31) - 15

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

    def mutate(self):
        """Mutate the last shape to create a mutation of this State.

        :return: mutated State instance
        :rtype: State
        """
        mutated_shape = self.rects[-1].mutate()

        # Copies the destination image, including its last rendering
        ndst = copy.copy(self.dst)

        nrects = copy.copy(self.rects)
        nrects[-1] = mutated_shape

        # Note: the new state is explicitly marked as mutation
        return State(src=self.src, dst=ndst, rects=nrects, _mutation=True)

    def finalize(self):
        """Finalize the current state, ending the mutation.

        :return: finalized State instance
        :rtype: State
        """
        assert self._mutation is True
        ndst = copy.copy(self.dst)
        nrects = copy.copy(self.rects)
        return State(src=self.src, dst=ndst, rects=nrects, _mutation=False)

    def error(self):
        """Compute current error against source reference.

        :return: error metric
        :rtype: int
        """
        return error(self.src, self.dst)

    def render(self):
        """Render the current state of the shapes.

        Raster version
        """
        # If the current state was just finalized after a mutation, there's no
        # need to draw anything: the destination picture is already a faithful
        # representation. Drawing anything would actually pollute the picture,
        # so we do an early exit.
        if self._mutation is False:
            return

        # If the current state is not a mutation, this means we can get away
        # with drawing only the last primitive - all the previous ones have
        # already been drawn into the destination image.
        if self._mutation is None:
            shapes = [self.rects[-1]]

        # Otherwise, we have to re-draw the whole state from scratch: if
        # _mutation is True, it means that this state is a direct mutation
        # of another one, so the image cache is invalid.
        else:
            assert self._mutation is True
            shapes = self.rects

        for shape in shapes:
            c = shape.col
            if c is None:
                mp = shape.center()
                mp = (clamp(0, mp[0], self.src.size[0] - 1), clamp(0, mp[1], self.src.size[1] - 1))

                pix = self.src.getpixel(mp)

                col = (pix[0], pix[1], pix[2], 0x7F)
            else:
                col = c

            shape.draw_pillow(self.imp, col)

    def dump_to_svg(self, filename):
        """Render the current state of the shapes.

        SVG version.
        """
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

                col = (pix[0], pix[1], pix[2], 0x7F)
            else:
                col = c

            shape.draw_svg(dwg, col)

        dwg.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="print information to stdout", action="count")
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
            if args.verbose > 0:
                print("Iter", a, "Error improved to", best_error)

        # Also run mutations on the best shape
        best_mutation_so_far = best_overall_so_far
        best_mutation_error = best_overall_error

        for m in range(100):

            ns = best_mutation_so_far.mutate()
            err = ns.error()
            if err < best_mutation_error:
                best_mutation_so_far = ns
                best_mutation_error = err
                if args.verbose > 1:
                    print("Mutation", m, "Error improved to", best_mutation_error)

        if best_mutation_error < best_overall_error:
            best_overall_error = best_mutation_error
            best_overall_so_far = best_mutation_so_far.finalize()

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

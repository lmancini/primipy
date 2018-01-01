"""Main primipy module."""

from __future__ import print_function

import argparse
import copy
import random

from PIL import Image
from PIL import ImageChops
from PIL import ImageDraw


def error(im1, im2):
    """Calculate the root-mean difference between two images."""
    im_i = ImageChops.difference(im1, im2)

    hist = im_i.histogram()

    h = zip(hist[:256], hist[256:512], hist[512:])
    area = float(im1.size[0]) * im1.size[1]

    err = sum((r + g + b) * (idx * idx) for idx, (r, g, b) in enumerate(h)) / area

    return err


def clamp(low, x, up):
    """Clamp x between low and up."""
    return min(max(low, x), up)


class State(object):
    def __init__(self, src, dst, rects=None):
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
            r = ('r', ((0, 0), (self.src.size[0] - 1, self.src.size[1] - 1)), avg_col)
            self.rects.append(r)
        else:
            # Assume client provides a copy
            self.rects = rects

        self.render()

    def randrect(self):
        maxw = self.dst.size[0]
        maxh = self.dst.size[1]

        x, y = random.randint(0, maxw), random.randint(0, maxh)
        w, h = random.randint(0, maxw), random.randint(0, maxh)

        w /= 2
        h /= 2

        p1 = (x - w / 2, y - h / 2)
        p2 = (x + w / 2, y + h / 2)
        return ('r', (p1, p2), None)

    def randtri(self):
        maxw = self.dst.size[0]
        maxh = self.dst.size[1]

        x1, x2, x3 = random.randint(0, maxw), random.randint(0, maxw), random.randint(0, maxw)
        y1, y2, y3 = random.randint(0, maxh), random.randint(0, maxh), random.randint(0, maxh)

        return ('t', ((x1, y1), (x2, y2), (x3, y3)), None)

    def improve(self):
        r = self.randtri()

        # Copies the destination image, including its last rendering
        ndst = copy.copy(self.dst)

        nrects = copy.copy(self.rects)
        nrects.append(r)

        return State(src=self.src, dst=ndst, rects=nrects)

    def error(self):
        return error(self.src, self.dst)

    def render(self):
        for (t, pts, c) in [self.rects[-1]]:
            if c is None:
                p1 = pts[0]
                p2 = pts[1]
                mp = (clamp(0, (p1[0] + p2[0]) / 2, self.src.size[0] - 1),
                      clamp(0, (p1[1] + p2[1]) / 2, self.src.size[1] - 1))

                pix = self.src.getpixel(mp)

                col = (pix[0], pix[1], pix[2], 0x77)
            else:
                col = c

            if t == 't':
                self.imp.polygon(pts, col)
            else:
                assert t == 'r'
                self.imp.rectangle(pts, col)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="print information to stdout", action="store_true")
    parser.add_argument("-i", dest="input", help="input image", required=True)
    parser.add_argument("-o", dest="output", help="output image", required=True)
    parser.add_argument("-n", dest="nshapes", type=int, help="number of shapes", required=True)
    parser.add_argument("-iters", dest="niters", type=int, help="number of iterations", default=100)

    args = parser.parse_args()

    im = Image.open(args.input).convert("RGB")

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
                print("Iter ", a, "Error improved to ", best_error)

        best_overall_so_far.dst.save(args.output)

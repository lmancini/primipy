"""Main primipy module."""

import math
import copy
import random

from PIL import Image
from PIL import ImageChops
from PIL import ImageDraw


def rmsdiff(im1, im2):
    """Calculate the root-mean-square difference between two images."""
    h = ImageChops.difference(im1, im2).histogram()

    return math.sqrt(sum(h * (i**2) for i, h in enumerate(h))) / (float(im1.size[0]) * im1.size[1])


def error(im1, im2):
    """Calculate the root-mean difference between two images."""
    im_i = ImageChops.difference(im1, im2)

    hist = im_i.histogram()

    h_r = hist[:256]
    h_g = hist[256:512]
    h_b = hist[512:]

    err_r = sum(r * (idx**2) for idx, r in enumerate(h_r)) / (float(im1.size[0]) * im1.size[1])
    err_g = sum(g * (idx**2) for idx, g in enumerate(h_g)) / (float(im1.size[0]) * im1.size[1])
    err_b = sum(b * (idx**2) for idx, b in enumerate(h_b)) / (float(im1.size[0]) * im1.size[1])

    return err_r + err_g + err_b


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
        else:
            # Assume client provides a copy
            self.rects = rects

    def improve(self, r):
        nrects = copy.copy(self.rects)
        nrects.append(r)
        return State(src=self.src, dst=self.dst, rects=nrects)

    def render(self):
        # Clear
        # hist = self.imp.im.histogram()

        # h_r = hist[:256]
        # h_g = hist[256:512]
        # h_b = hist[512:]

        # mr = sum(r * idx for idx, r in enumerate(h_r)) / sum(h_r)
        # mg = sum(g * idx for idx, g in enumerate(h_g)) / sum(h_g)
        # mb = sum(b * idx for idx, b in enumerate(h_b)) / sum(h_b)

        # Ok just kidding
        mr = mg = mb = 0

        self.imp.rectangle([(0, 0), (self.src.size[0] - 1, self.src.size[1] - 1)], (mr, mg, mb, 0x77))
        for (p1, p2) in self.rects:
            mp = (clamp(0, (p1[0] + p2[0]) / 2, self.src.size[0] - 1),
                  clamp(0, (p1[1] + p2[1]) / 2, self.src.size[1] - 1))

            pix = self.src.getpixel(mp)
            p3 = (p1[0], p2[1])
            self.imp.polygon([p1, p2, p3], (pix[0], pix[1], pix[2], 0x77))


def randrect(maxw, maxh):
    x, y = random.randint(0, maxw), random.randint(0, maxh)
    w, h = random.randint(0, maxw), random.randint(0, maxh)

    w /= 2
    h /= 2

    return [(x - w / 2, y - h / 2), (x + w / 2, y + h / 2)]

if __name__ == '__main__':
    im = Image.open("../imgs/ddo.png")

    im2 = Image.new("RGB", (im.width, im.height))

    state = State(im, im2)

    best_overall_so_far = state
    best_overall_error = rmsdiff(im, im2)
    # Polygons in the image
    for a in range(100):

        best_so_far = None
        best_error = None

        # New polygons to try
        for b in range(100):

            ns = best_overall_so_far.improve(randrect(im.size[0], im.size[1]))
            ns.render()

            if best_so_far is None:
                best_so_far = ns
                best_error = rmsdiff(im, im2)
                continue

            err = rmsdiff(im, im2)
            if err < best_error and err < best_overall_error:
                best_so_far = ns
                best_error = err

        if best_error < best_overall_error:
            best_overall_error = best_error
            best_overall_so_far = best_so_far
            im2.save("r.png")
            print "switch!", best_error, a

        print best_overall_error

"""Main primipy module."""

import copy
import random
import sys

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
        else:
            # Assume client provides a copy
            self.rects = rects

            self.render()

    def improve(self, r):
        # Copies the destination image, including its last rendering
        ndst = copy.copy(self.dst)

        nrects = copy.copy(self.rects)
        nrects.append(r)

        return State(src=self.src, dst=ndst, rects=nrects)

    def error(self):
        return error(self.src, self.dst)

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
        # mr = mg = mb = 0

        # self.imp.rectangle([(0, 0), (self.src.size[0] - 1, self.src.size[1] - 1)], (mr, mg, mb, 0x77))
        for (p1, p2) in [self.rects[-1]]:
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
    im = Image.open(sys.argv[1]).convert("RGB")

    im2 = Image.new("RGB", (im.width, im.height))

    state = State(im, im2)

    best_overall_so_far = state
    best_overall_error = state.error()
    # Polygons in the image
    for a in range(100):

        best_so_far = None
        best_error = None

        # New polygons to try
        for b in range(100):

            ns = best_overall_so_far.improve(randrect(im.size[0], im.size[1]))

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
            print "switch!", best_error, a

        print best_overall_error

        best_overall_so_far.dst.save("tmp2.png")

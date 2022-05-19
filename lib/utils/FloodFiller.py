import sys
from PIL import Image

class ProgressBar(object):

    """Docstring for ProgressBar. TODO"""

    def __init__(self, size, barLength=40):
        """TODO: to be defined1.
        :size: TODO
        """
        self.size = size
        self.barLength = barLength

    def update(self, n):
        self._drawProgressBar(n / self.size, self.barLength)


    @staticmethod
    def _drawProgressBar(percent_done, barLength):
        """Display an updating progress bar in a terminal
        :percent_done: the percent done to display
        :barLength: how many chars long the bar is
        :returns: None
        """
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLength):
            if i <= int(barLength * percent_done):
                if i+1 <= int(barLength * percent_done):
                    progress += "="
                else:
                    progress += ">"
            else:
                progress += " "
        sys.stdout.write("[%s] %.2f%%" % (progress, percent_done * 100))
        sys.stdout.flush()
class _Filter(object):

    """Base Class for a Filter"""

    def __init__(self, im, filter_matrix=None, show_progress_bar=True):
        """Init for a base filter
        :im:            the image
        :filter_matrix: the matrix to apply to each pixel
        :returns:       None
        """
        self.im = im
        self.height = im.shape[0]

        self.width = im.shape[1]

        self.filter_matrix = filter_matrix
        self.progress_bar = ProgressBar(self.height)
        self.show_progress_bar = show_progress_bar


    def update(self, n):
        if self.show_progress_bar:
            self.progress_bar.update(n+1)


    def new_image(self):
        ret = []
        for y in range(self.height):
            ret.append([0] * self.width)
        return ret


    @staticmethod
    def _average(s, n):
        return sum(s) / n


    @staticmethod
    def filter_range(x):
        return range(-(x//2), x-(x//2))

class FloodFiller(_Filter):

    """Docstring for FloodFiller. TODO"""


    def __init__(self, im, growth, defaults={}, show_progress_bar=True):
        """TODO: to be defined1.
        :im: TODO
        :growth: TODO
        :defaults: TODO
        :show_progress_bar: TODO
        """
        _Filter.__init__(self, im, show_progress_bar=show_progress_bar)
        self.defaults = defaults
        self.growth = growth


    def fill(self):
        #TODO make object id generation not crappy
        #for now just assign a high value
        least = 9999
        n = least
        for y in range(self.height):
            self.update(y)
            for x in range(self.width):
                if self.im[y][x] < least:
                    if self.im[y][x] in self.defaults:
                        self._floodfill(x, y, self.defaults[ self.im[y][x] ])
                    elif self.im[y][x] not in self.defaults.values():
                        self._floodfill(x, y, n)
                        n += 1


    def _floodfill(self, x, y, number):
        """Does an iterative flood fill starting at spot x,y """
        val = self.im[y][x]
        q = []
        q.append((x, y))
        while len(q) > 0:
            x, y = q.pop()
            self.im[y][x] = number
            for i in range(-self.growth, self.growth+1):
                for j in range(-self.growth, self.growth+1):
                    if i == 0 and j == 0:
                        continue
                    if y+j < 0 or y+j >= self.height or x+i < 0 or x+i >= self.width or self.im[y+j][x+i] != val:
                        continue
                    q.append((x+i, y+j))


    def to_image(self, color_map=None, color_wheel=None):
        if color_map is None:
            color_map = {
                    7: (255, 255, 255),
                    8: (0, 0, 0),
                    9: (255, 255, 0),
                    }
        if color_wheel is None:
            color_wheel = [
                    (255, 0, 255),
                    (0, 255, 255),
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 0, 125),
                    (255, 125, 0),
                    (125, 0, 255),
                    (125, 255, 0),
                    (255, 255, 125),
                    (255, 125, 255),
                    (125, 255, 255),
                    (255, 125, 75),
                    (255, 75, 125),
                    (125, 255, 75),
                    (125, 75, 255),
                    (75, 125, 255),
                    (75, 255, 125),
                    ]

        image = Image.new('RGBA', (self.width, self.height), 'BLACK')
        color_counter = 0
        for y in range(self.height):
            self.update(y)
            for x in range(self.width):
                if self.im[y][x] not in color_map:
                    color_map[self.im[y][x]] = color_wheel[color_counter]
                    color_counter = (color_counter + 1) % len(color_wheel)
                image.putpixel((x, y), color_map[self.im[y][x]])

        return image


class EdgeGrower(object):

    """Docstring for EdgeGrower. TODO"""

    def __init__(self, im, show_progress_bar=True):
        """TODO: to be defined1.
        :im: TODO
        """
        self._im = im
        self.show_progress_bar = show_progress_bar


    def update(self, n):
        if self.show_progress_bar:
            self.progress_bar.update(n+1)


    def grow(self, threshold):
        pass


class SimpleBlurFilter(_Filter):

    """Docstring for SimpleBlurFilter. TODO"""

    def __init__(self, im, filter_matrix, threshold=None, show_progress_bar=True):
        """Init for SimpleBlurFilter
        :im: TODO
        :filter_matrix: TODO
        """
        _Filter.__init__(self, im, filter_matrix, show_progress_bar=show_progress_bar)
        self.threshold = threshold


    def filter(self, display_averages=False):
        filter_length = len(self.filter_matrix)
        filter_total = filter_length * filter_length
        half = filter_length // 2
        new_image = self.new_image()

        for y in range(half, self.height-half):
            self.update(y+half)
            for x in range(half, self.width-half):
                total = 0
                for a in _Filter.filter_range(filter_length):
                    for b in _Filter.filter_range(filter_length):
                        total += self.im[y+a][x+b] * self.filter_matrix[a+half][b+half]
                total = round(total/filter_total)
                if display_averages:
                    print("%d" % (total))
                if self.threshold is not None:
                    if total >= self.threshold:
                        new_image[y][x] = 255
                else:
                    new_image[y][x] = total

        return new_image

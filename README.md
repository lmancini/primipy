# Primipy

Artistic reproduction of images with geometric shapes.

![Cafe beans](https://lmancini.github.io/static/primipy/cafe.jpg)

### What is it?

Primipy is a command-line application that reads a reference image, and produces a set of geometric shapes that approximate the reference when drawn in sequence.

The algorithm adds one shape at a time. Each one goes through randomization and optimization steps. In detail:

 1. An arbitrary random number of shapes are tested, and the one that minimizes the error against the reference is chosen (randomization);
 2. The chosen shape is locally optimized via vertex mutations, and the best mutation is chosen (optimization).

### Usage

Install Python 2.x, then

```
pip install -r requirements.txt
python main.py -i <your_reference_image> -o <output file> -n 200 -v
```

A SVG with vector shapes will be produced in the same directory of the output file.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `i` | n/a | input file |
| `o` | n/a | output file |
| `n` | n/a | number of shapes |
| `iters` | 100 | number of iteration (randomization step) |
| `v` | False | verbose output |
| `vv` | False | more verbose output |

### Extras

If using PNG as output format, Primipy stores some information in the final output, like how many shapes were used, and other useful data. Imagemagick's `identify` utility can be used to inspect this information:

```
identify -verbose <image.png>
```

### Motivation

This was inspired by the Primitive program by Michael Fogleman (https://github.com/fogleman/primitive/). I was fascinated by the beautiful results produced by such a randomized process and decided to explore it in more depth.

### Gallery

![ladybug](https://lmancini.github.io/static/primipy/ladybug.jpg)
![cat](https://lmancini.github.io/static/primipy/cat.jpg)
![tomatoes](https://lmancini.github.io/static/primipy/tomatoes.jpg)
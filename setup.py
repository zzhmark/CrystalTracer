from setuptools import setup, Extension
import numpy


setup(
    ext_modules=[
        Extension(
            'crystal_tracer.algorithm.detection',
            ['crystal_tracer/algorithm/detection.pyx'],
            language='c++',
            include_dirs=[numpy.get_include()],
            ),
        Extension(
            'crystal_tracer.visual.draw',
            ['crystal_tracer/visual/draw.pyx'],
            language='c++',
            include_dirs=[numpy.get_include()],
            ),
        Extension(
            "crystal_tracer.algorithm.gwdt.gwdt_impl",
            ["crystal_tracer/algorithm/gwdt/gwdt_impl.pyx"],
            language="c++",
            include_dirs=[numpy.get_include()]
        )
    ]
)


from setuptools import setup, Extension
import numpy

setup(
    name="HexaVidCounter",
    version='0.1',
    author='zzh',
    ext_modules=[
        Extension(
            'tracer.detection',
            sources=['tracer/detection.pyx'],
            language='c++',
            include_dirs=[numpy.get_include()],
            library_dirs=[],
            libraries=[],
            extra_compile_args=[],
            extra_link_args=[]
            ),
        Extension(
            'tracer.draw',
            sources=['tracer/draw.pyx'],
            language='c++',
            include_dirs=[numpy.get_include()],
            library_dirs=[],
            libraries=[],
            extra_compile_args=[],
            extra_link_args=[]
            ),
        Extension(
            name="gwdt.gwdt_impl",
            sources=["gwdt/gwdt_impl.pyx"],
            include_dirs=[numpy.get_include()],
            language="c++"
        )
    ]
)

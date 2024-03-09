from setuptools import setup

setup(
    name='StyleTTS2',
    version='0.0.0',
    packages=['StyleTTS2'],
    install_requires=[
        'soundfile==0.12.1',
        'torchaudio==2.1.2+cu118',
        'munch==4.0.0',
        'torch==2.1.2+cu118',
        'pydub==0.25.1',
        'pyyaml==6.0.1',
        'librosa==0.10.1',
        'nltk==3.8.1',
        'matplotlib==3.8.2',
        'accelerate==0.26.1',
        'transformers==4.37.0',
        'einops==0.7.0',
        'einops-exts==0.0.4',
        'tqdm==4.66.1',
        'typing==3.7.4.3',
        'typing-extensions==4.9.0',
        'monotonic-align @ git+https://github.com/resemble-ai/monotonic_align.git@78b985be210a03d08bc3acc01c4df0442105366f'
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu118'
    ],
    author='See URL for code authors, packaging by Sam Nason-Tomaszewski',
    url='https://github.com/yl4579/StyleTTS2',
    description='See URL for full description. This fork is from the 9cb642e47e9c0adddcd2763d06468e787cfc1494 commit.'
)
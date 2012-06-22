from setuptools import setup, find_packages

setup(name="pyOpenTLD",
    version = 0.1,
    download_url = "https://github.com/jayrambhia/pyOpenTLD/downloads/tarball/master",
    description = "Python port of openTLD",
    keywords = "opencv, cv, simplecv, opentld, tracking, median flow, lucas kanede",
    author = "Jay Rambhia",
    author_email = "jayrambhia777@gmail.com",
    license = 'BSD',
    packages = find_packages(),
    requires = ["cv2","cv","simplecv"],
    scripts = ["scripts/opentld"]
    )

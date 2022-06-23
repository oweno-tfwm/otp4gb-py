from urllib.request import urlretrieve
from subprocess import run


def setup():
    urlretrieve('http://m.m.i24.cc/osmconvert64.exe', 'bin/osmconvert64.exe')
    run('mvn dependency:copy-dependencies copy-rename:rename', shell=True)

if __name__ == '__main__':
    setup()
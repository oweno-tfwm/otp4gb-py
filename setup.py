import zipfile
from urllib.request import urlretrieve
from subprocess import run


def setup():
    urlretrieve('http://m.m.i24.cc/osmconvert64.exe', 'bin/osmconvert64.exe')
    urlretrieve('https://dlcdn.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.zip', 'bin/maven.zip')
    with zipfile.ZipFile('bin/maven.zip', 'r') as zip_ref:
      zip_ref.extractall('bin')
    run('mvn dependency:copy-dependencies copy-rename:rename', shell=True)

if __name__ == '__main__':
    setup()
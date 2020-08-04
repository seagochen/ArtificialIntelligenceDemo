import os
import sys

from siki.basics import FileUtils
from siki.basics import Exceptions


class Switch(object):

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:
            self.fall = True
            return True
        else:
            return False


def unzip_files(folder):
    if not FileUtils.exists(folder):
        raise Exceptions.NoAvailableResourcesFoundException(f"{folder} does not exist.")

    zip_files = FileUtils.search_files(folder, r"\.zip$")

    # iterate every file and unzip it
    for f in zip_files:
        os.system(f"unzip {f}")


def untar_files(folder):
    if not FileUtils.exists(folder):
        raise Exceptions.NoAvailableResourcesFoundException(f"{folder} does not exist.")

    tar_files = FileUtils.search_files(folder, r"\.tar$")

    # iterate every file and unzip it
    for f in tar_files:
        os.system(f"tar xvf {f}")


def unxz_files(folder):
    if not FileUtils.exists(folder):
        raise Exceptions.NoAvailableResourcesFoundException(f"{folder} does not exist.")

    tar_files = FileUtils.search_files(folder, r"\.xz")

    # iterate every file and unzip it
    for f in tar_files:
        os.system(f"xz -d {f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("no available folder path")
        exit(0)

    for case in Switch(sys.argv[1]):

        if case('unzip'):
            unzip_files(sys.argv[2])
            break

        if case('untar'):
            untar_files(sys.argv[2])
            break

        if case('unxz'):
            unxz_files(sys.argv[2])
            break

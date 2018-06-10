import os
import sys
import time
from urllib.request import urlretrieve


def main(source_filename):
    cloud_kind = source_filename.split(".")[0]
    dirname = "./" + cloud_kind
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for filespec in open(source_filename):
        number, rest, *__ = filespec.split(";;")
        rest = rest.replace("_m.", ".")
        url = ("https://cloudatlas.wmo.int/images/compressed/" + number +
               "_main_" + rest)
        local_filename = dirname + "/" + number + "_" + rest
        print("downloading image " + number)
        urlretrieve(url, local_filename)
        time.sleep(0.5)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("please specify a filename from which we can infer image URLs")
        sys.exit(1)

    main(sys.argv[1])

import os
import time
from urllib.request import urlretrieve


data = {
    "cirrus": {
        "fibratus": 14,
        "uncinus": 7,
        "spissatus": 9,
        "castellanus": 1,
        "floccus": 3,
        "intortus": 7,
        "vertebratus": 11,
        "radiatus": 3,
        "duplicatus": 2,
        "mamma": 2,
    },
    "cirrostratus": {
        "fibratus": 3,
        "nebulosus": 1,
        "undulatus": 6,
    },
    # "cirrocumulus": {
    #     "castellanus": 1,
    #     "floccus": 13,
    #     "stratiformis": 1,
    #     "undulatus": 8,
    #     "lacunosus": 2,
    #     "mamma": 3,
    #     "virga": 6,
    # },
    "altocumulus": {
        "castellanus": 1,
        "floccus": 8,
        "stratiformis": 6,
        "lenticularis": 20,
        "undulatus": 7,
        "radiatus": 1,
        "duplicatus": 5,
        "perlucidus": 6,
        "translucidus": 5,
        "mamma": 4,
        "virga": 14,
    },
    # "altostratus": {
    #     "virga": 2,
    #     "mamma": 1,
    #     "radiatus": 2,
    #     "undulatus": 2,
    #     "translucidus": 5,
    #     "duplicatus": 2,
    #     "pannus": 1,
    #     "opacus": 1,
    # },
    # "stratocumulus": {
    #     "undulatus": 15,
    #     "lacunosus": 3,
    #     "radiatus": 3,
    #     "lenticularis": 4,
    #     "mamma": 3,
    #     "stratiformis": 2,
    #     "virga": 2,
    #     "perlucidus": 3,
    #     "castellanus": 2,
    #     "translucidus": 1,
    #     "opacus": 1,
    # },
    # "stratus": {
    #     "fractus": 5,
    #     "nebulosus": 6,
    #     "opacus": 2,
    # },
    "nimbostratus": {
        "pannus": 4,
        "praecipitatio": 3,
    },
    "cumulus": {
        "velum": 2,
        "pileus": 7,
        "radiatus": 4,
        "congestus": 19,
        "mediocris": 8,
        "arcus": 1,
        "virga": 3,
        "pannus": 1,
        "humilis": 2,
        "praecipitatio": 1,
    },
    "cumulonimbus": {
        "mamma": 23,
        "arcus": 23,
        "incus": 19,
        "calvus": 10,
        "praecipitatio": 9,
        "pileus": 7,
        "capillatus": 7,
        "tuba": 6,
        "virga": 3,
        "pannus": 3,
    },
}


for kind, subkinds in data.items():
    dirname = "./clouds_online/" + kind
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for subkind, num_images in subkinds.items():
        for x in range(1, num_images + 1):
            url = ("http://www.wolken-online.de/images/wolken/" + kind + "/" +
                   kind + "_" + subkind)

            if x > 1:
                url += "_" + str(x)

            url += ".jpg"
            local_filename = dirname + "/" + kind + subkind + str(x)

            try:
                print("downloading image " + url)
                urlretrieve(url, local_filename)
                time.sleep(0.5)
            except:
                print("FAILED: " + url)

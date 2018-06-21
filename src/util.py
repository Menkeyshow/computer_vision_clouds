import os

def ensure_directory_exists(dirname):
    """Checks if the directory `dirname` exists, and creates it if it
    doesn't.

    Returns `True` if the directory existed already, and `False` if it
    had to be created."""

    if os.path.exists(dirname):
        return True
    else:
        os.makedirs(dirname)
        return False


cloud_kinds = {
    "stratiform":
        ["cirrostratus", "altostratus", "nimbostratus", "stratus"],
    "cirriform": ["cirrus"],
    "stratocumuliform": ["cirrocumulus", "altocumulus", "stratocumulus"],
    "cumuliform": ["cumulus", "cumulonimbus"]
}

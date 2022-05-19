import os
import socket
from datetime import datetime


def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()

def get_output_dir(args, run_name):
    """ Get root output directory for each run """
    cfg_filename, _ = os.path.splitext(os.path.split(args.cfg_file)[1])
    return os.path.join(args.output_base_dir, cfg_filename, run_name)

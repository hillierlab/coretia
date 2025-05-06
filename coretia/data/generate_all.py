import os
from coretia.data import run


if __name__ == "__main__":
    # Replace 'base_directory' with the actual path where your folders are located
    base_directory = os.path.dirname(os.path.abspath(__file__))
    run(base_directory)

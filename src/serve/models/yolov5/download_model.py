import typer
from yolov5.utils.downloads import attempt_download


def main(file):
    """"""
    attempt_download(file)


if __name__ == "__main__":
    typer.run(main)

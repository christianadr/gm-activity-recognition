from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytube import YouTube
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Downloading video clips from YouTube")
    parser.add_argument(
        "--src",
        help="source list of video clips to download in .txt format",
        required=True,
    )
    parser.add_argument(
        "--dst",
        help="target directory where to save videos",
        required=True,
    )

    return parser.parse_args()


def download_video(url: str, dst: Path):
    """Download a single video from a given url

    Parameters
    ----------
    url : str
        URL of the YouTube clip to download
    dst : Path
        Destination directory where to save video
    """
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(dst)
    except Exception as e:
        print(f"Failed to download {url}: {e}")


def main():
    # TODO: Not yet tested

    args = parse_args()

    src, dst = Path(args.src), Path(args.dst)

    if not dst.exists():
        dst.mkdir(parents=True)

    with src.open("r") as file:
        urls = file.readlines()

    for url in tqdm(urls, total=len(urls), desc="Saving YouTube clips"):
        url = url.strip()
        if url:
            download_video(url, dst)


if __name__ == "__main__":
    main()

import argparse


def get_argparse(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        epilog='If there are any issues or questions, feel free to visit our GitHub repository at '
               'https://github.com/gigl-carleton/ByExampleSynthesisOfVectorTextures'
    )
    return parser

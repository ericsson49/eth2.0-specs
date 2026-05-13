import argparse
from pathlib import Path

from .py_to_mzn import Convertor


def main() -> None:
    parser = argparse.ArgumentParser(description="Transpile a Python constraint model to MiniZinc")
    parser.add_argument("input", type=Path, help="Python model to transpile")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="MiniZinc output path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    convertor = Convertor()
    output = convertor.convert(args.input.read_text())

    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output)


if __name__ == "__main__":
    main()

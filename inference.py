import click

from model import Dcscn


@click.command()
@click.argument("input")
def main(input):
    model = Dcscn()
    model.inference(input)


if __name__ == "__main__":
    main()

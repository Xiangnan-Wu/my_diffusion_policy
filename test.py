import click

@click.command()
@click.option('--a', type=int, required=True)
@click.option('--b', type=int, required=True)
def main(a, b):
    print(a + b)

if __name__ == "__main__":
    main()
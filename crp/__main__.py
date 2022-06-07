import click


@click.group()
def cli():
    print(r"""
   ___ ___ ___ 
  / __| _ \ _ \
 | (__|   /  _/
  \___|_|_\_|  
               
""")


@click.command()
@click.argument('path')  # help='path to filename that contains the CRP inferface')
def analyze(path):
    """
    Run Feature Visualization.

    PATH points to filename that contains the CRP interface
    """

    print("FV running")


@click.command()
@click.argument('path')  # help='path to textfile that contains the paths to checkpoints')
def collect(path):

    print("collcet running")


cli.add_command(analyze)
cli.add_command(collect)

if __name__ == '__main__':
    cli()

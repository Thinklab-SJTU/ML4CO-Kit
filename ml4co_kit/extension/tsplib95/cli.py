# -*- coding: utf-8 -*-
"""Console script for tsplib95."""
import click
from tabulate import tabulate

from . import loaders
from . import exceptions


DEFAULT_COLUMNS = ['DIMENSION', 'TYPE', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT']  # noqa: E501


def load(filepaths):
    files = {}
    with click.progressbar(filepaths, label='Loading files',
                           show_pos=True, show_eta=True) as bar:
        for filepath in bar:
            try:
                file = loaders.load(filepath)
            except exceptions.ParsingError as e:
                error = f'\n{filepath}: {e}'
                click.secho(error, fg='red')
            else:
                files[filepath] = file

    return files


def get_tabular_data(files, columns=None, by_keyword=True):
    columns = columns or DEFAULT_COLUMNS

    def get_values(data):
        for column in columns:
            yield data[column]

    rows = []
    for filepath, file in files.items():
        values = get_values(file.as_dict(by_keyword=by_keyword))
        rows.append([filepath] + list(values))

    header = ['file'] + list(columns)
    return header, rows


@click.group()
def cli():
    pass


@cli.command()
@click.argument('filepaths', nargs=-1,
                type=click.Path(dir_okay=False, readable=True))
@click.option('-c', '--column', 'columns', multiple=True,
              default=DEFAULT_COLUMNS)
@click.option('--by-keyword/--by-name', default=True)
def summarize(filepaths, columns, by_keyword):
    """Console script for tsplib95."""
    if not filepaths:
        return

    header, rows = get_tabular_data(load(filepaths),
                                    columns=columns,
                                    by_keyword=by_keyword)
    click.echo_via_pager(tabulate(sorted(rows), header))


if __name__ == '__main__':
    cli()

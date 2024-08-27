from datetime import datetime
import json
import os
import numpy as np
import pathlib
from pandas.api.types import is_float_dtype
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from pprint import pprint
# import PIL
# import plotly.express as px


def join_path(a, *p): return os.path.abspath(os.path.join(a, *p))


def print_json(name='dic', dic={}, index=None):
    if index is not None:
        keys = [*dic]
        dic = {keys[i]: dic[keys[i]] for i in index}
    dic_str = json.dumps(
        obj=dic,
        default=lambda o: ' '.join(repr(o).split()),
        indent=4
    )
    print(f'{name}: \n{dic_str}')


def add_suffix(filepath, suffix):
    """This function adds suffix to file name before its extension.
    Example:
        add_suffix('path/dir/foo.txt', '_1')
        -> 'path/dir/foo_1.txt'
    """
    folder_path, filename = os.path.split(filepath)
    filename, extension = os.path.splitext(filename)
    newfilename = filename + suffix + extension
    newpath = os.path.join(folder_path, newfilename)

    return newpath


def distinct_filename(filepath):
    """If file already exists, a _i suffix is added.
    """
    newpath = filepath
    is_file = os.path.isfile(newpath)
    i = 1
    while is_file:
        newpath = add_suffix(filepath, f'_{i}')
        is_file = os.path.isfile(newpath)
        i += 1
    filepath = newpath
    return filepath


def save(fig, name, width=4, path=None, image_ext='png', check_exist=False):
    """
    Exemple of use:
        save(
            fig=fig,
            name=kwargs['fig_name'],
            width=4,
            path=kwargs['path'],
            image_ext='png',
            check_exist=False
        )

    If check_exist=True, if fig_save_path is already a file, another suffix is
    added to distinct new fig_save_path from existing one.

    If path last folder does nots exists, mkdir is done for it but not its
    parents.
    """
    if path is None:
        path = os.path.abspath(os.path.join(
            os.path.abspath(__file__),
            '..', '..', '..', 'investigation_plots',
        ))
    fig_save_path = join_path(
        path,
        name
    )
    # Create subdir if needed
    pathlib.Path(path).mkdir(
        parents=False,
        exist_ok=True
    )
    # Check if file already exists if check_exist set to True
    fig_save_path_image = f'{fig_save_path}.{image_ext}'
    fig_save_path_html = f'{fig_save_path}.html'
    if check_exist:
        fig_save_path_image = distinct_filename(fig_save_path_image)
        fig_save_path_html = distinct_filename(fig_save_path_html)

    print(f'Save plot: {os.path.splitext(fig_save_path_image)[0]}')
    fig.write_image(
        file=fig_save_path_image,
        format=image_ext,
        width=width*300,
        height=2*300,
        scale=3
    )
    fig.write_html(
        file=fig_save_path_html
    )
    # fig.show()


def fill_dict(keys, value):
    """create a new dict with 'keys' as keys and fill each one with
    'value'.
    """
    return {e: value for e in keys}


def format_customdata(i, col, is_float):
    """
    Returns string of hover message depending if data is numerical.
    If is_float, return scientific format
        <b>col</b>: %{customdata[i]:.2e}
    Else, return raw
        <b>col</b>: %{customdata[i]}
    """
    x = 'customdata[' + str(i) + ']'
    if is_float:
        str_out = (
            f'<b>{col}</b>: '
            + '%{' + x + ':.2e}'
        )
    else:
        str_out = (
            f'<b>{col}</b>: '
            + '%{' + x + '}'
        )
    return str_out


def hover_df(custom_cols, float_cols, i_mainvals=2):
    """
    Arguments
    ---------

    custom_cols: list
        List of strings of all columns to use for hover
    float_cols: list
        List of strings of only numerical columns to use for hover
    i_mainvals: int
        Number of rows before empty line separation.

    Return
    ------
    hover: string
        Hover message with empty separation line when more than 2
        exemple:
            <b>x</b>: %{customdata[0]:.2e}<br>
            <b>y</b>: %{customdata[1]:.2e}<br>
            <br>
            <b>info_1</b>: %{customdata[2]}
            <b>info_2</b>: %{customdata[3]:.2e}
            <b>info_3</b>: %{customdata[4]:.2e}
    """
    rows = list(map(
        lambda x: format_customdata(
            i=x[0],
            col=x[1],
            is_float=(x[1] in float_cols)
        ),
        enumerate(custom_cols)
    ))
    if len(custom_cols) > i_mainvals:
        # add empty spearation line
        rows = rows[:i_mainvals] + [''] + rows[i_mainvals:]
    hover_str = '<br>'.join(rows)

    return hover_str


def list_files(dir, extensions=''):
    """
    Extract all available result images directly in dir
    If nested image, will not be found
    """
    if extensions == 'images':
        extensions = (
            'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif', 'svg', 'html'
        )
    img_paths = []
    for file in os.listdir(dir):
        if file.endswith(extensions):
            img_paths.append(os.path.join(dir, file))
    return img_paths


def plot_line(df, **kwargs):
    """
    Returns a fig with x, y and color axis

    Arguments
    ---------
    df : pandas.DataFrame
        the currently visible entries of the datatable
    x, y_columns: str, 
        name of the datatable column that is used to define x-axis
    y: str or List of str
        name of the datatable column(s) that is(are) used to define y-axis

    Optional kwargs
    ---------------
    hover_cols: list
        Columns to use for hover info
    color_column: str
        name of the datatable column that is used to define colorscale of the overlay
    mode: str or Dict with y_columns as keys
        Any combination of ['lines', 'markers', 'text'] joined with '+' characters
        (e.g. 'lines+markers')

    Example
    -------
    fig = plot_line(
        df=df,
        x=x_column,
        y=y_column,
        color_column=color_column,
        title=f'{y_column}({x_column})',
        legend_title="",
        y_hover={y_column: hover_cols},
        annotations=(
            f'(x, y, color): {tuple(df[axes].count())}'
        ),
    )
    
    Code snippet
    ------------
    import numpy as np
    import pandas as pd
    
    x_values = np.linspace(-1, 1, 100)
    x_values = np.concatenate((x_values[x_values<0], [0], x_values[x_values>0]))
    x_abs_max = np.max(np.abs(x_values))
    y_lin = x_values**2
    y_abs_sqrt = np.sqrt(np.abs(x_values))
    y_exp = x_values*np.exp(x_values)

    df = pd.DataFrame({
        'x': x_values,
        'y_lin': y_lin,
        'y_abs_sqrt': y_abs_sqrt,
        'y_x_exp': y_exp,
    })
    fig = plot_line(df=df, x='x', y=['y_lin', 'y_abs_sqrt', 'y_x_exp'], mode='lines')
    fig.show()
    """

    if 'x' not in [*kwargs]:
        assert False, "Missing column name (string) for x-axis"
    if 'y' not in [*kwargs]:
        assert False, "Missing column name (string or list of strings) for y-axis"
    if isinstance(kwargs['y'], str):
        kwargs['y'] = [kwargs['y']]
    if 'y_name' not in [*kwargs]:
        kwargs['y_name'] = {}
    if 'y_hover' not in [*kwargs]:
        kwargs['y_hover'] = {}
    if 'title' not in [*kwargs]:
        kwargs['title'] = f'{kwargs['y']}({kwargs['x']})'
        kwargs['title'] += f'<br> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    if 'legend_title' not in [*kwargs]:
        kwargs['legend_title'] = ''
    if 'mode' not in [*kwargs]:
        kwargs['mode'] = 'markers'

    # Add traces
    traces = []

    x = kwargs['x']
    color_list = [kwargs['color_column']] if 'color_column' in kwargs else []
    for y in kwargs['y']:
        hover_cols = kwargs['y_hover'].get(y, [])
        default_cols = [x, y] + color_list
        hover_cols = [col for col in hover_cols if col not in default_cols]
        hover_cols = default_cols + hover_cols
        float_cols = [col for col in hover_cols if is_float_dtype(df[col])]
        df_hover = df.fillna('NaN')
        traces.append(
            dict(
                x=df.loc[:, x],
                y=df.loc[:, y],
                name=kwargs['y_name'].get(y, y),
                showlegend=True,
                # line=dict(
                #     dash='solid',
                #     width=2,
                # ),
                mode=kwargs['mode'],
                customdata=df_hover.loc[:, hover_cols],
                hovertemplate=hover_df(
                    custom_cols=hover_cols,
                    float_cols=float_cols,
                    i_mainvals=len(default_cols),
                )
            )
        )
    if 'color_column' in kwargs:
        traces[0].update(  # this is only done for first y trace
            dict(
                marker=dict(
                    color=df.loc[:, kwargs['color_column']],
                    showscale=True,
                    colorscale='Jet',  # Inferno, Bluered, Rainbow, Turbo
                    colorbar=dict(
                        title=dict(
                            text=kwargs['color_column'],
                            font=dict(
                                # size=axis_titles_font_size,
                                family='Serif',
                            ),
                        ),
                        titleside='right'  # vertical title angle
                    ),
                )
            )
        )
    layout = dict(
        title=dict(
            text=kwargs['title'],
            y=0.96,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        legend_title_text=kwargs['legend_title'],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis=dict(
            title=kwargs.get('x_title', kwargs['x']),
        ),
        yaxis=dict(
            title=kwargs.get('y_title', '')
        ),
        template='plotly_white',
    )

    # print('\n traces:')
    # pprint(
    #     object=traces,
    #     indent=4,
    #     sort_dicts=False
    # )
    # print('\n layout:')
    # pprint(
    #     object=layout,
    #     indent=4,
    #     sort_dicts=False
    # )

    fig = go.Figure(
        data=[
            *list(map(go.Scatter, traces)),
        ],
        layout=go.Layout(layout)
    )
    if 'annotations' in kwargs:
        fig.add_annotation(
            text=kwargs['annotations'].replace('\n', '<br>'),
            x=0., xref='paper', xanchor='left',
            y=1., yref='paper', yanchor='bottom',
            align='left',  # left, center
            showarrow=False,
        )
    return fig


def plot_gp_std_2d(model, X_train: np.ndarray, target_idx=0, z_value=None, plot_type='surface'):
    """
    Plot the standard deviation of the Gaussian Process for a grid of points between 0 and 1.

    Parameters:
    - model: Trained Gaussian Process model.
    - X_train (np.ndarray): Training data.
    - target_idx (int): Index of the target variable to plot.
    - z_value (float): Fixed value for the third dimension. If None, defaults to the mean of the first two training points.
    - plot_type (str): Type of plot ('surface' or 'contour').
    """
    n = 100
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n, n))

    # Determine z_value if not provided
    if z_value is None:
        z_value = X_train[:, 2].mean()

    # Calculate standard deviation for each grid point
    for i in range(n):
        for j in range(n):
            _, y_std = model.predict(np.array([[X[i, j], Y[i, j], z_value]]), return_std=True)
            Z[i, j] = y_std[0, target_idx]

    # Create the plot
    if plot_type == 'surface':
        fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
        # Add training points
        x_train_2d = X_train[:, :2]
        dummy_train = np.hstack((x_train_2d, np.full((x_train_2d.shape[0], 1), z_value)))
        _, y_std_train = model.predict(dummy_train, return_std=True)
        z_train = y_std_train[:, target_idx]
        fig.add_trace(go.Scatter3d(
            x=x_train_2d[:, 0], y=x_train_2d[:, 1], z=z_train, mode='markers',
            marker=dict(color='green', size=3), name='Training Points'
        ))
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Standard Deviation'))
    elif plot_type == 'contour':
        fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, contours=dict(coloring='heatmap')))
        fig.update_layout(coloraxis=dict(colorscale='Viridis'))

    fig.update_layout(legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01))
    fig.update_layout(height=800, margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

    print("Done plotting.")


# if __name__ == "__main__":
#     test things
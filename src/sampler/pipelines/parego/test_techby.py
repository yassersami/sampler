import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sampler.pipelines.sao_optim.plotly_utils import plot_line
from sampler.pipelines.parego.nodes import LambdaGenerator, tchebychev

def set_2d_space():
    targets = ['Pg_f', 'Tg_Tmax']
    lambda_gen = LambdaGenerator(k=len(targets), s=1000)
    llambda = lambda_gen.choose_uniform_lambda()

    x_values = np.linspace(0, 1, 10)
    y_values = np.linspace(0, 1, 10)
    x, y = np.meshgrid(x_values, y_values)  # (x[i], y[i]) form lines in (x, y) plane 
    z = np.array([tchebychev(np.array([x_i, y_i]).T, llambda) for x_i, y_i in zip(x, y)])

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(x, y, z)
    ax.set_xlabel(f'f_1: {targets[0]}')
    ax.set_ylabel(f'f_2: {targets[1]}')
    ax.set_zlabel('f_llambda')
    ax.set_title(f'llambda(1, 2): {llambda}')
    plt.show()

    # fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    # fig.update_layout(
    #     title=f'Tchebychev f_llambda on [0, 1]²',
    #     # autosize=False, width=500, height=500,
    #     # margin=dict(l=65, r=50, b=65, t=90)
    # )
    # fig.show()

set_2d_space()
import numpy as np
import pandas as pd
import sampler.pipelines.sao_optim.nodes as sao_optim_nodes
from sampler.pipelines.sao_optim.plotly_utils import plot_line


def parallel_jobs():
    from multiprocessing import Process, Queue
    q = Queue()
    q.put(5)
    q.put(q.get()+5)
    print(f'Final q: {q.get()}')


def test_bandpass_filter():
    L = 2.0
    U = 5.0
    dist = U - L
    bounds_ratio = 7
    x_values = np.linspace(L-dist*bounds_ratio, U+dist*bounds_ratio, 1000)
    # Ensure L and U are in the x array
    if L not in x_values:
        x_values = np.concatenate((x_values[x_values < L], [L], x_values[x_values >= L]))
    if U not in x_values:
        x_values = np.concatenate((x_values[x_values <= U], [U], x_values[x_values > U]))
    # Compute f(x)
    y_lin_tent = sao_optim_nodes.linear_tent(x_values, L, U, slope=1)
    y_sigmoid_tent_UL = sao_optim_nodes.sigmoid_tent(x_values, L, U, k=1/(U-L))
    y_sigmoid_tent_8UL = sao_optim_nodes.sigmoid_tent(x_values, L, U, k=8 * np.log(2) / (U - L))
    y_gaussian_tent = sao_optim_nodes.gaussian_tent(x_values, L, U, sigma=1)
    y_gaussian_tent_10 = sao_optim_nodes.gaussian_tent(x_values, L, U, sigma=10)
    y_gaussian_tent_UL = sao_optim_nodes.gaussian_tent(x_values, L, U, sigma=(U - L) / np.sqrt(8 * np.log(2)))
    # Prepare for plot
    df = pd.DataFrame({
        "x": x_values,
        "y_lin_tent": y_lin_tent,
        "y_sigmoid_tent_UL": y_sigmoid_tent_UL,
        "y_sigmoid_tent_8UL": y_sigmoid_tent_8UL,
        "y_gaussian_tent": y_gaussian_tent,
        "y_gaussian_tent_10": y_gaussian_tent_10,
        "y_gaussian_tent_UL": y_gaussian_tent_UL,
    })
    fig = plot_line(
        df=df, x="x", mode='lines',
        y=[
            # "y_lin_tent",
            "y_sigmoid_tent_UL",
            "y_sigmoid_tent_8UL",
            # "y_gaussian_tent",
            # "y_gaussian_tent_10",
            "y_gaussian_tent_UL",
        ]
    )
    fig.show()

if __name__ == "__main__":
    # test_bandpass_filter()
    parallel_jobs()
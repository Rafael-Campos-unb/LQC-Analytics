import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

esc = StandardScaler()


def centroids_begin(k, data):
    centroids = data.sample(k)
    return centroids


np.random.seed(42)


# Root square error
def rser(a, b):
    return np.square(np.sum((a - b) ** 2))


# Correlating clusters with dataset
def centroid_attributes(data, centroids):
    k = centroids.shape[0]
    n = data.shape[0]
    attribution = []
    attributes_error = []
    # Estimating error:
    for i in range(n):
        tot_errors = np.array([])
        for centroid in range(k):
            err = rser(centroids.iloc[centroid, :], data.iloc[i, :])
            tot_errors = np.append(tot_errors, err)
        nearest_centroid = np.where(tot_errors == np.amin(tot_errors))[0].tolist()[0]
        nearest_centroid_error = np.amin(tot_errors)
        attribution.append(nearest_centroid)
        attributes_error.append(nearest_centroid_error)
    return attribution, attributes_error


# df_esc_T_corr['cluster'], df_esc_T_corr['error'] = centroid_attributes(df_esc_T_corr, centroids)
# centroids = df_esc_T_corr.groupby('cluster').agg('mean').loc[:, df_cols].reset_index(drop=True)
def kmeans(data, k, lim=1e-4):
    data_copy = data.copy()
    err = []
    go_on = True
    j = 0
    centroids = centroids_begin(k, data)
    while go_on:
        data_copy['cluster'], j_err = centroid_attributes(data_copy, centroids)
        err.append(sum(j_err))
        centroids = data_copy.groupby('cluster').agg('mean').reset_index(drop='True')
        if j > 0:
            if err[j - 1] - err[j] <= lim:
                go_on = False
        j += 1
    data_copy['cluster'], j_err = centroid_attributes(data_copy, centroids)
    centroids = data_copy.groupby('cluster').agg('mean').reset_index(drop=True)
    fig2 = px.scatter(data_copy, data.index[0], data.index[-1],
                      labels={
                          # data.index[0]: f"Feature space of {data.index[0]}",
                          # data.index[-1]: f"Feature space of {data.index[-1]}"
                          data.index[0]: f"Espaço de característica do {data.index[0]}",
                          data.index[-1]: f"Espaço de característica do {data.index[-1]}"
                      },
                      color='cluster',
                      symbol=data.index)
    fig2.layout.coloraxis.colorbar.title = 'cluster'
    colors_clusts = [i for i in range(0, k + 1)]
    fig2.update_layout(showlegend=False)
    fig2.update_layout(font=dict(
        family='Arial Black',
        size=18
    ))
    fig2.add_traces(
        go.Scatter(x=centroids.iloc[:, 0], y=centroids.iloc[:, 1], mode='markers', marker=dict(color=colors_clusts),
                   marker_symbol='circle', marker_size=25, name='centroid', opacity=0.2))
    return fig2, data_copy['cluster'], j_err


def elbow_graph(data):
    err_total = []
    n = 10
    elbow = data[data.index]
    for i in range(n):
        _, _, j_err = kmeans(elbow, i + 1)
        err_total.append(sum(j_err))
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(range(1, n + 1), err_total, linewidth=3, marker='o')
    ax.set_xlabel(r'Number of clusters (n)', fontsize=14)
    ax.set_ylabel(r'WSS/Inertia', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

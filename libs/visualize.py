import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis_points_3d(points, save_name):
    # print(pos_events.shape)
    points = points.to("cpu").detach().numpy()
    fig = plt.figure()
    ax = Axes3D(fig)
    # fig = fig.add_subplot(111, projection="3d")
    ax.plot(
        points[:, 2],
        points[:, 0],
        points[:, 1],
        c="red",
        marker="o",
        linestyle="None",
    )

    plt.savefig(save_name)
    plt.close(fig)


def vis_points_2d(points, save_name):
    # print(pos_events.shape)
    points = points.to("cpu").detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(
        points[:, 0],
        points[:, 1],
        c="red",
        marker="o",
        linestyle="None",
    )

    plt.savefig(save_name)
    plt.close(fig)

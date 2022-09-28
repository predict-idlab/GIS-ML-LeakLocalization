"""Map plotter that can plot a hydraulic network and its nodes with a value attached to the nodes as a heat map."""

import sys
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from adjustText import adjust_text


def draw_network(g, coords, directory, name,
                 colors=None, sizes=None, errors=None,
                 color_nodes=False):
    global cmap
    texts = []
    if errors is not None:
        cmap = plt.get_cmap('YlOrRd')
        norm = Normalize(vmin=0, vmax=1)

    def drawLineBetween(ax, NodeId1, NodeId2):
        coords1 = coords[NodeId1]
        coords2 = coords[NodeId2]
        if (coords2 is not None) and (coords1 is not None):
            xx = [coords1[0], coords2[0]]
            yy = [coords1[1], coords2[1]]
            color = 'gray'
            width = 2
            if errors is not None and not color_nodes:
                color = cmap(0)
                if NodeId1 in errors and NodeId2 in errors:
                    color = cmap(norm((errors[NodeId1] + errors[NodeId2]) / 2))
                if (NodeId1, NodeId2) in errors:
                    color = cmap(norm(errors[(NodeId1, NodeId2)]))
            ax.plot(xx, yy, color=color, linewidth=width, zorder=1)
            return xx, yy

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all', sharey='all')

    xs, ys, cs, ss = [], [], [], []
    for key in coords.keys():
        color = colors[key]
        if errors is not None and color_nodes:
            if color == 'black':
                color = cmap(0)
                if key in errors:
                    color = cmap(norm(errors[key]))
        xs.append(coords[key][0])
        ys.append(coords[key][1])
        cs.append(color)
        ss.append(sizes[key])

    if not color_nodes:
        new_xs, new_ys, new_cs, new_ss = [], [], [], []
        for i, c in enumerate(cs):
            if c != 'black':
                new_xs.append(xs[i])
                new_ys.append(ys[i])
                new_cs.append(cs[i])
                new_ss.append(ss[i])
        plt.scatter(x=new_xs, y=new_ys, c=new_cs, s=new_ss, zorder=2)
    else:
        plt.scatter(x=xs, y=ys, c=cs, s=ss, zorder=2)

    minx, miny, maxx, maxy = sys.maxsize, sys.maxsize, -sys.maxsize, -sys.maxsize
    for i, e in enumerate(g.edges()):
        if e[0] in coords and e[1] in coords:
            xx, yy = drawLineBetween(ax, e[0], e[1])
            minx = min(min(minx, xx[0]), xx[1])
            miny = min(min(miny, yy[0]), yy[1])
            maxx = max(max(maxx, xx[0]), xx[1])
            maxy = max(max(maxy, yy[0]), yy[1])
    adjust_text(texts, only_move={'texts': 'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.box(on=None)
    plt.axis('off')

    print("saving fig... " + str(directory) + "/network_" + str(name) + ".png")
    plt.savefig(str(directory) + "/network_" + str(name) + ".png", dpi=300)
    plt.close()

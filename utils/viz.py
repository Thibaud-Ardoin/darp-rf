from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
from .constants import DEPOT_TNODE
from mpl_toolkits.mplot3d import proj3d

plt.style.use('seaborn-paper')

COLOR_DIVERGING_DARK_10 = [[228, 26, 28],
                           [55, 126, 184],
                           [77, 175, 74],
                           [152, 78, 163],
                           [255, 127, 0],
                           [255, 255, 51],
                           [166, 86, 40],
                           [247, 129, 191],
                           [153, 153, 153]]
COLOR_DIVERGING_DARK_10 = [[v / 255 for v in c] for c in COLOR_DIVERGING_DARK_10]

class Arrow3D(matplotlib.patches.FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(self, renderer)

class BaseNetwork:
    EDGE_COLORMAP = np.array(COLOR_DIVERGING_DARK_10)

    def __init__(self):
        self._color_idx = 0

    def get_next_color(self):
        c = self.EDGE_COLORMAP[self._color_idx]
        self._color_idx = (self._color_idx + 1) % len(self.EDGE_COLORMAP)
        return c


class TimedNetwork(BaseNetwork):
    TNODE_COLOR = (0.6, 0.6, 0.6, 1)
    TNODE_END_COLOR = COLOR_DIVERGING_DARK_10[0]
    ARC_COLOR = COLOR_DIVERGING_DARK_10[1]
    FRAGMENT_COLOR = COLOR_DIVERGING_DARK_10[2]
    BORDER_COLOR = (0, 0, 0, 0)
    TNODE_PLOT_OPTS = dict(marker='o',markersize=.5, linewidth=0.5)
    EDGE_PLOT_OPTS = dict(marker='o',markersize=.5, linewidth=0.5)

    def __init__(self, timed_nodes, data, hide_unused=True):
        super().__init__()
        tnodes = set(timed_nodes) - {DEPOT_TNODE}
        self.hide_unused = hide_unused
        self._node_xpos = None
        self.data = data
        self.paths = []
        self.num_nodes = len(set(i for i, t in tnodes))

        node_times = defaultdict(list)
        for i, t in tnodes:
            node_times[i].append(t)

        self.node_times = dict()
        self.node_time_window = dict()
        min_time = np.inf
        max_time = -np.inf

        for i, times in node_times.items():
            self.node_times[i] = sorted(times)
            self.node_time_window[i] = [data.E[i], data.L[i]]
            min_time = min(min_time, *self.node_times[i], *self.node_time_window[i])
            max_time = max(max_time, *self.node_times[i], *self.node_time_window[i])

        self.ylim = [min_time - 0.05 * (max_time - min_time), max_time + 0.05 * (max_time - min_time)]
        self.node_width = None


    @property
    def node_xpos(self):
        if self._node_xpos is None:
            self._node_xpos = dict()
            paths = [p for p, _, _ in self.paths]
            paths.sort(key=len)
            xshift = list(map(len, paths))
            xshift = sum(xshift[:-1]) / 2 + xshift[-1]
            xshift = 1 / xshift
            x = 0
            for i, p in enumerate(paths):
                path_width = 0
                for node in p:
                    if node not in self._node_xpos:
                        self._node_xpos[node] = x
                        x += xshift
                        path_width += xshift
                x -= path_width / 2

            self.node_width = xshift / 2
        return self._node_xpos


    def plot(self):
        fig, ax = plt.subplots()
        for i, x in self.node_xpos.items():
            Y = self.node_times[i]
            for y in Y:
                ax.plot([x, x + self.node_width], [y, y], color=self.TNODE_COLOR, **self.TNODE_PLOT_OPTS)
            for y in self.node_time_window[i]:
                ax.plot([x, x + self.node_width], [y, y], color=self.TNODE_END_COLOR, **self.TNODE_PLOT_OPTS)

            ax.annotate('{:2d}{:s}'.format(((i-1)%self.data.n)+1, 'D' if i > self.data.n else 'P'),
                        ha='center', va='bottom', fontsize=3,
                        xy=(x + self.node_width / 2, self.node_time_window[i][1]))

        for p, tarcs, tfragments in self.paths:
            # col = self.get_next_color()

            for ti, tj in tarcs:
                if ti == DEPOT_TNODE:
                    start_x, start_y = self.node_xpos[tj[0]], self.ylim[0]
                else:
                    start_x, start_y = self.node_xpos[ti[0]] + self.node_width, ti[1]

                if tj == DEPOT_TNODE:
                    end_x, end_y = self.node_xpos[ti[0]] + self.node_width, self.ylim[1]
                else:
                    end_x, end_y = self.node_xpos[tj[0]], tj[1]

                ax.plot([start_x, end_x], [start_y, end_y], linewidth=0.5, marker=None, color=self.ARC_COLOR)

            for tfragment in tfragments:
                x = []
                y = []
                for ti, tj in zip(tfragment[:-1], tfragment[1:]):
                    x.extend([self.node_xpos[ti[0]] + self.node_width, self.node_xpos[tj[0]]])
                    y.extend([ti[1], tj[1]])
                ax.plot(x, y, color=self.FRAGMENT_COLOR, **self.EDGE_PLOT_OPTS)


        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim(self.ylim)
        ax.set_xticks([])
        ax.set_ylabel('Time')
        plt.show()


    def add_path(self, tnode_seq, tarcs, filled_tfragments):
        node_seq = list(OrderedDict((tnode[0], None) for tnode in tnode_seq if tnode != DEPOT_TNODE).keys())
        self.paths.append([node_seq, tarcs, filled_tfragments])
        self._node_xpos = None


class Network(BaseNetwork):
    def __init__(self, node_loc, edges=None, hide_unused=True):
        super().__init__()
        a = node_loc.min()
        b = node_loc.max()
        self.node_loc = (node_loc - a) / (b - a)
        self.edges = set(edges) if edges is not None else None
        self.edge_colors = defaultdict(list)
        self.edge_dir = defaultdict(list)
        self.paths = []
        self.hide_unused = hide_unused
        self.node_mask = np.zeros(self.node_loc.shape[0], dtype=np.bool)

    def add_path(self, path, fmt_str=None):
        path_color = self.get_next_color()
        for p in zip(path[:-1], path[1:]):
            if len(p) != 2:
                raise ValueError('`path` should be an iterable of 2-tuples')
            elif self.edges is not None and p not in self.edges:
                raise ValueError('the edge {} -> {} does not exist'.format_map(p))

            k = tuple(sorted(p))
            self.edge_colors[k].append(path_color)
            self.edge_dir[k].append(p[0] > p[1])
        self.node_mask[path] = True
        self.paths.append(path)

    def plot(self):
        fig, ax = plt.subplots(dpi=200)

        for arc, colors in self.edge_colors.items():

            swap_direction = self.edge_dir[arc]
            start_x = self.node_loc[arc[0], 0]
            start_y = self.node_loc[arc[0], 1]
            end_x = self.node_loc[arc[1], 0]
            end_y = self.node_loc[arc[1], 1]

            kwargs = []

            sgn = 1
            d_rad = 0.1
            if len(colors) % 2 == 0:
                rad = d_rad / 2
                inc_rad_condn = lambda i: i % 2 == 1
            else:
                rad = 0
                inc_rad_condn = lambda i: i % 2 == 0

            for i, c in enumerate(colors):
                mod_sgn = 1 - int(swap_direction[i]) * 2

                kw = {
                    'connectionstyle': matplotlib.patches.ConnectionStyle.Arc3(rad=rad * sgn * mod_sgn),
                    'color': c
                }

                if swap_direction[i]:
                    kw['posA'] = (end_x, end_y)
                    kw['posB'] = (start_x, start_y)
                else:
                    kw['posA'] = (start_x, start_y)
                    kw['posB'] = (end_x, end_y)

                kwargs.append(kw)

                if inc_rad_condn(i):
                    rad += d_rad
                else:
                    sgn *= -1

            for kw in kwargs:
                ax.add_patch(matplotlib.patches.FancyArrowPatch(
                    shrinkB=12,
                    shrinkA=12,
                    arrowstyle=matplotlib.patches.ArrowStyle.CurveB(head_length=4, head_width=1),
                    **kw
                ))

        for i in range(self.node_loc.shape[0]):
            if self.node_mask[i]:
                ax.add_patch(plt.Circle(self.node_loc[i, :], radius=0.05, fill=False))
                ax.annotate(f"{i:d}", ha='center', va='center', xy=self.node_loc[i, :], fontsize=12)

        ax.set_aspect('equal')
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        plt.show()

#!/usr/bin/env python3

import copy
from collections import defaultdict
from functools import reduce
import itertools
import operator
from typing import Dict, List, Set, Tuple

import igraph as ig
from igraph import GraphBase, Graph, Vertex
import matplotlib.pyplot as plt

JOBS = 8

Triangle = Set[Vertex]

AdjMatrix = List[List[int]]

################ Representation of tile runs ################


class TileRun:
    # memory optimization
    __slots__ = ("vs", "adj", "names", "frame")

    def __init__(self, vs: int, adj: AdjMatrix, names: List[str], frame: str):
        self.vs = vs
        self.adj = adj
        # names may not be unique, but lt, lb, rt, rb uniquely identify the 4 corners of the tile run.
        self.names = names
        self.frame = frame

    @classmethod
    def from_encoding(cls, encoding: str) -> "TileRun":
        """Construct a tile run from its encoding."""
        assert encoding[-1] == "L", "Invalid encoding"

        decoded = list(map(lambda x: cls._decode(x + "L"), encoding.split("L")[:-1]))

        assert len(decoded) % 2 == 1, "Encoding must consist of odd number of tiles."
        assert len(decoded) >= 5, "Encoding must consist of at least 5 tiles."

        return reduce(operator.add, map(cls._construct_tile, decoded))

    def to_graph(self) -> Graph:
        """Convert this object to an igraph.Graph"""
        g = Graph.Adjacency(self.adj, mode="undirected")
        g.vs["names"] = self.names
        return g

    def to_2cc_graph(self) -> Graph:
        """Convert this object to an igraph.Graph, then connect the ends to construct a 2cc graph proper."""
        self = copy.deepcopy(self)

        # connect corners of the tile run
        lt, lb, rt, rb = self._get_corners()

        if self.frame == "L":
            self.adj[lb][rt], self.adj[rt][lb] = 1, 1
            contract_edge(self.adj, ["lb", "rt"], lb, rt)
        else:
            self.adj[lb][rt], self.adj[rt][lb] = 2, 2

        self.adj[lt][rb], self.adj[rb][lt] = 1, 1

        g = Graph.Adjacency(self.adj, mode="undirected")
        g.vs["names"] = self.names
        return g

    @classmethod
    def all_tiles(cls) -> Dict[str, "TileRun"]:
        """Return all tiles (Mapping from encoding to TileRun object)"""
        return {
            encoding: cls._construct_tile(cls._decode(encoding))
            for encoding in all_tile_encodings()
        }

    @classmethod
    def all_runs(cls, runlen: int) -> Dict[str, "TileRun"]:
        """Return all tile runs of a certain length (Mapping from encoding to TileRun object)"""
        combinations = itertools.combinations_with_replacement(
            cls.all_tiles().items(), runlen
        )
        combinations = map(lambda x: tuple(zip(*x)), combinations)
        combinations = map(
            lambda x: ("".join(x[0]), reduce(operator.add, x[1])), combinations
        )
        return dict(combinations)

    def __iadd__(self, other) -> "TileRun":
        """Glue two tile runs together to produce a new tile run (destroying old ones)"""
        adj1, adj2 = self.adj, other.adj
        n1, n2 = len(adj1), len(adj2)

        adj = adj1

        for i, _ in enumerate(adj):
            adj[i] += [0] * n2

        for row in adj2:
            adj.append([0] * n1 + row)

        # lt ----------- rt <---> lb ------------- rb
        # |              |        |                |
        # | Tile 1       |        | Tile 2         |
        # |              |        |                |
        # lb ----------- rb <---> lt ------------- rt

        rtc = self.names.index("rt")
        ltc = other.names.index("lb") + n1

        rbc = self.names.index("rb")
        lbc = other.names.index("lt") + n1

        # rename everything so we can glue more shit together
        self.names[rtc] = "irt"
        other.names[ltc - n1] = "ilb"
        self.names[rbc] = "irb"
        other.names[lbc - n1] = "ilt"

        # connect everything up
        adj[rtc][ltc], adj[ltc][rtc] = 1, 1
        adj[rbc][lbc], adj[lbc][rbc] = 1, 1

        names = self.names + other.names

        frame1, frame2 = self.frame, other.frame

        if frame1 == "dL":
            # tile 1 has dL frame, add loop.
            adj[rtc][ltc], adj[ltc][rtc] = 2, 2
        else:
            # tile 1 has L frame, we need to contract edge rt <-> rb
            adj = contract_edge(adj, names, rtc, ltc)

        frame = frame2

        return type(self)(n1 + n2, adj, names, frame)

    def __add__(self, other) -> "TileRun":
        """Glue two tile runs together to produce a new tile run (without destroying old one)"""

        # deepcopy so we don't mess with originals.
        self = copy.deepcopy(self)
        other = copy.deepcopy(other)

        self += other

        return self

    ### low level fuckery to construct tiles: users should ignore this!

    FEATURES = {
        "D": {
            "vs": 2,
            "adj": [[0, 2], [2, 0]],
            "top": ["lt", "rt"],
            "bot": ["lb", "rb"],
        },
        "A": {
            "vs": 3,
            "adj": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            "top": ["lt", "l", "rt"],
            "bot": ["rb", "r", "lb"],
        },
        "V": {
            "vs": 3,
            "adj": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            "top": ["lt", "rt", "r"],
            "bot": ["rb", "lb", "l"],
        },
        "B": {
            "vs": 4,
            "adj": [[0, 2, 0, 0], [2, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]],
            "top": ["lt", "t", "rt", "r"],
            "bot": ["rb", "b", "lb", "l"],
        },
    }

    @staticmethod
    def _decode(encoding: str) -> Tuple[str, str, str, str]:

        top, encoding = encoding[0], encoding[1:]
        contract, bot = "", ""

        if top != "H":
            assert top in ["A", "V", "B", "D"], "Invalid encoding."
            if encoding[0] == "I":
                contract, encoding = encoding[0], encoding[1:]

            bot, encoding = encoding[0], encoding[1:]
            assert bot in ["A", "V", "B", "D"], "Invalid encoding."

            assert not contract or (top, bot) in [
                ("A", "B"),
                ("A", "V"),
                ("V", "A"),
                ("B", "A"),
            ], "Invalid encoding."

        assert encoding in ["L", "dL"], "Invalid encoding."

        return (top, contract, bot, encoding)

    @classmethod
    def _construct_tile(cls, tilespec: Tuple[str, str, str, str]) -> "TileRun":
        top, contract, bot, frame = tilespec

        if top == "H":
            return cls(
                6,
                [
                    [0, 1, 0, 1, 0, 0],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0],
                ],
                ["lt", "l", "lb", "rt", "r", "rb"],
                frame,
            )
        else:

            def get_connecting_vertex(
                vnames: List[str], isright: bool, top: bool
            ) -> int:
                try:
                    return vnames.index("r" if isright else "l")
                except ValueError:
                    return vnames.index(
                        ("r" if isright else "l") + ("t" if top else "b")
                    )

            topf, botf = cls.FEATURES[top], cls.FEATURES[bot]
            top_vnames, bot_vnames = topf["top"], botf["bot"]

            adj1, adj2 = topf["adj"], botf["adj"]
            n1 = len(adj1)
            n2 = len(adj2)

            adj = copy.deepcopy(adj1)

            for i, _ in enumerate(adj):
                adj[i] += [0] * n2

            for row in adj2:
                adj.append([0] * n1 + row)

            # connect everything up
            ltc = get_connecting_vertex(top_vnames, False, True)
            lbc = get_connecting_vertex(bot_vnames, False, False) + n1
            rtc = get_connecting_vertex(top_vnames, True, True)
            rbc = get_connecting_vertex(bot_vnames, True, False) + n1

            adj[ltc][lbc], adj[lbc][ltc] = 1, 1
            adj[rtc][rbc], adj[rbc][rtc] = 1, 1

            names = topf["top"] + botf["bot"]

            if contract:
                if "l" in top_vnames:
                    u, v = [i for i, v in enumerate(names) if v == "l"]
                    # contraction happens along left wall
                    adj = contract_edge(adj, names, u, v)
                else:
                    u, v = [i for i, v in enumerate(names) if v == "r"]
                    adj = contract_edge(adj, names, u, v)

            return cls(n1 + n2, adj, names, frame)

    def _get_corners(self) -> Tuple[int, int, int, int]:
        return tuple(map(self.names.index, ["lt", "lb", "rt", "rb"]))


############ Computing isolation number ################


def vertex_triangles(g: GraphBase) -> Dict[Vertex, Set[Triangle]]:
    vt = defaultdict(set)
    for triangle in g.cliques(min=3, max=3):
        for v in triangle:
            # sorting the tuples ensure that e.g. (a, b, c) and (b, a, c) are considered to be the same triangle, as they should.
            vt[v].add(tuple(sorted(triangle)))

    return vt


def neighbourhood(g: GraphBase, v: Vertex) -> Set[Vertex]:
    return set(g.neighbors(v)) | {v}


def vertex_neighbourhood_triangles(g: GraphBase) -> Dict[Vertex, Set[Triangle]]:
    vt = vertex_triangles(g)
    return {
        v: set().union(*[vt[neighbour] for neighbour in neighbourhood(g, v)])
        for v in g.vs.indices
    }


def c3_isolation_number_upper_bound(g: GraphBase) -> Tuple[Set[Vertex], int]:
    """Return upper bound on C3 isolation number, as well as an isolating set of this size."""
    g = copy.deepcopy(g)

    isolation_number = 0
    isolating_set = set()
    while (
        not all(map(set().__eq__, (vt := vertex_neighbourhood_triangles(g)).values()))
        and vt != dict()  # deal with special case of no triangles
    ):
        # while there are triangles, remove the neighbourhood belonging to largest number of triangles.
        v = max(vt.items(), key=lambda x: len(x[1]))[0]
        g.delete_vertices(neighbourhood(g, v))
        isolating_set |= {v}
        isolation_number += 1
    return isolating_set, isolation_number


def is_triangle_free(g: GraphBase) -> bool:
    return len(g.cliques(min=3, max=3)) == 0


def c3_isolation_number(g: GraphBase) -> Tuple[Set[Vertex], int]:
    """Exact C3 isolation number calc., returns minimal isolating set as well.
    Brute force algorithm: slow for large g.
    """

    potential_isolating_vertices = set(
        v
        for v, triangles in vertex_neighbourhood_triangles(g).items()
        if triangles != set()
    )

    for i in range(min(len(potential_isolating_vertices), len(g.vs.indices) // 4) + 1):
        for iset in itertools.combinations(potential_isolating_vertices, i):
            g_ = copy.deepcopy(g)
            g_.delete_vertices(
                set().union(*list(map(lambda v: neighbourhood(g, v), iset)))
            )
            if is_triangle_free(g_):
                return set(iset), i
    raise RuntimeError(g.get_adjacency())


################ Periphery functions ################


def all_tile_encodings() -> List[str]:
    bases = [
        "DD",
        "DV",
        "DB",
        "DA",
        "VV",
        "AIB",
        "AV",
        "VB",
        "VA",
        "VIA",
        "BB",
        "BV",
        "AB",
        "BD",
        "BA",
        "BIA",
        "AA",
        "H",
        "AD",
        "AIV",
        "VD",
    ]

    return list(map(lambda x: x + "L", bases)) + list(map(lambda x: x + "dL", bases))


def plot_all_tiles(fname: str = "tiles.svg"):
    """Plot all the tiles."""
    rows, cols = 6, 7
    _, axs = plt.subplots(rows, cols, figsize=(8, 4))
    for i, t in enumerate(TileRun.all_tiles().values()):
        ig.plot(
            t.to_graph(),
            target=axs[i % rows, i // rows],
        )
    plt.savefig(fname)


################ low level adjacency matrix code ################
# Because igraph can't contract edges.


def contract_edge(adj: AdjMatrix, names: List[str], u: int, v: int) -> AdjMatrix:
    for i, n in enumerate(adj[v]):
        adj[u][i] += n

    adj.pop(v)

    for i, _ in enumerate(adj):
        adj[i][u] += adj[i].pop(v)

    adj[u][u] -= 2

    names.pop(v)
    return adj


################ Application code ################


def maximize_c3_isolation_number_per_n_minus_1(
    runs: Dict[str, TileRun]
) -> Tuple[str, float]:
    """Find a tile run amongst runs which has the largest i(G, K_3)/(n-1)."""
    maximal_encoding, max_isolation_number_per_n_minus_1 = max(
        {
            encoding: c3_isolation_number(g)[1] / (len(g.vs.indices) - 1)
            for encoding, g in map(lambda x: (x[0], x[1].to_graph()), runs.items())
        }.items(),
        key=lambda x: x[1],
    )
    return maximal_encoding, max_isolation_number_per_n_minus_1


if __name__ == "__main__":
    # find the run of 2 tiles which maximizes i(G, K_3) / (n - 1) (and the maximal value)
    print(*maximize_c3_isolation_number_per_n_minus_1(TileRun.all_runs(2)))

    # find the run of 3 tiles which maximizes i(G, K_3) / (n - 1) (and the maximal value)
    print(*maximize_c3_isolation_number_per_n_minus_1(TileRun.all_runs(3)))

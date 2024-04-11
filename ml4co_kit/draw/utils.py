import networkx as nx


SUPPORT_POS_TYPE_DICT = {
    "bipartite_layout": nx.bipartite_layout,
    "circular_layout": nx.circular_layout,
    "kamada_kawai_layout": nx.kamada_kawai_layout,
    "random_layout": nx.random_layout,
    "rescale_layout": nx.rescale_layout,
    "rescale_layout_dict": nx.rescale_layout_dict,
    "shell_layout": nx.shell_layout,
    "spring_layout": nx.spring_layout,
    "spectral_layout": nx.spectral_layout,
    "planar_layout": nx.planar_layout,
    "fruchterman_reingold_layout": nx.fruchterman_reingold_layout,
    "spiral_layout": nx.spiral_layout,
    "multipartite_layout": nx.multipartite_layout,
}

SUPPORT_POS_TYPE = [
    "bipartite_layout",
    "circular_layout",
    "kamada_kawai_layout",
    "random_layout",
    "rescale_layout",
    "rescale_layout_dict",
    "shell_layout",
    "spring_layout",
    "spectral_layout",
    "planar_layout",
    "fruchterman_reingold_layout",
    "spiral_layout",
    "multipartite_layout",
]


def get_pos_layer(pos_type: str):
    if pos_type not in SUPPORT_POS_TYPE:
        raise ValueError(f"unvalid pos type, only supports {SUPPORT_POS_TYPE}")
    return SUPPORT_POS_TYPE_DICT[pos_type]

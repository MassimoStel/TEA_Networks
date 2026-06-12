import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from teanets.nlp_utils import compute_valence
from teanets.analytics import svo_to_graph


def plot_svo_graph(df, subject_filter=None, object_filter=None, custom_font=14, filename=None, mark_passive_approx=False, seed=None, show=True):
    """
    Plot a graph of SVO data.

    Args:
        df (dataframe): a pandas DataFrame of SVO data.
        subject_filter (str): A subject to filter the graph by.
        object_filter (str): An object to filter the graph by.
        custom_font (int): Font size for node labels.
        filename (str): If set, save the figure to this path.
        mark_passive_approx (bool): If True, edges that originate exclusively
            from passive constructions without an explicit agent
            (``passive_approx == 1``) are drawn with a dashed style and
            reduced alpha to visually distinguish "victim-in-Agent"
            approximations from genuine perpetrator edges.
            Default ``False`` preserves the legacy rendering.
        seed (int): Seed for the (random) node layout. Set it to obtain
            reproducible figures.
        show (bool): If True (default), display the figure; if False, close
            it after the optional save (useful in batch/headless runs).
    """
    G = svo_to_graph(df, subject_filter=subject_filter, object_filter=object_filter)
    plot_graph(G, custom_font, filename, mark_passive_approx=mark_passive_approx, seed=seed, show=show)


def plot_graph(G, custom_font, filename=None, mark_passive_approx=False, seed=None, show=True):
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        print("plot_graph: the graph is empty, nothing to plot.")
        return

    rng = np.random.default_rng(seed)
    # Scale figure size based on number of nodes
    figsize = (
        max(12, 8 + (num_nodes**0.7) * 1.4),
        max(7.5, 4.5 + (num_nodes**0.7) * 1.25),
    )
    plt.figure(figsize=figsize)

    # Separate nodes by type
    subjects = [
        node
        for node, attr in G.nodes(data=True)
        if "subject" in attr.get("type", set())
    ]
    verbs = [
        node for node, attr in G.nodes(data=True) if "verb" in attr.get("type", set())
    ]
    objects = [
        node for node, attr in G.nodes(data=True) if "object" in attr.get("type", set())
    ]

    # Compute valences first
    valences = {}
    node_colors = {}
    for node in G.nodes():
        label = G.nodes[node].get("label", node)
        valence = compute_valence(label)
        valences[node] = valence
        if valence == "positive":
            node_colors[node] = "#1f77b4"  # Blue
        elif valence == "negative":
            node_colors[node] = "#d62728"  # Red
        else:
            node_colors[node] = "#7f7f7f"  # Grey

    # Find connected components
    sentences = []
    for v in verbs:
        s_nodes = set()
        for s in subjects:
            if G.has_edge(s, v):
                s_nodes.add(s)
        o_nodes = set()
        for o in objects:
            if G.has_edge(v, o):
                o_nodes.add(o)
        if s_nodes or o_nodes:
            sentences.append((s_nodes, {v}, o_nodes))

    # Position calculation with increased spacing
    pos = {}
    spacing = 2  # Increased from 1.2 for more space between subgraphs

    def get_available_position(
        base_x, base_y, existing_positions, x_range=0.45, tolerance=0.85
    ):
        x = base_x + rng.uniform(-x_range, x_range)
        y = base_y
        attempts = 0
        while attempts < 50 and any(
            abs(ex_y - y) < tolerance and abs(ex_x - x) < tolerance
            for ex_x, ex_y in existing_positions.values()
        ):
            x = base_x + rng.uniform(-x_range, x_range)
            y += spacing * 0.3  # Increased from 0.25 for more vertical spacing
            attempts += 1
        return x, y

    # Initial positioning with more spread between sentences
    sentence_base_y = -2
    for s_nodes, v_nodes, o_nodes in sentences:
        # Increase vertical gap between subgraphs
        sentence_base_y += (
            spacing * 1.2
        )  # Increased multiplier for more space between subgraphs

        # Add some random variation to prevent perfect alignment
        base_y = sentence_base_y + rng.uniform(-0.2, 0.2)

        # Position subjects with more vertical spread
        for s in s_nodes:
            x, y = get_available_position(
                -2, base_y + rng.uniform(-0.4, 0.4), pos
            )  # Increased range
            pos[s] = (x, y)

        # Position verbs
        for v in v_nodes:
            x, y = get_available_position(0, base_y, pos, x_range=0.25)
            pos[v] = (x, y)

        # Position objects with more vertical spread
        for o in o_nodes:
            x, y = get_available_position(
                2, base_y + rng.uniform(-0.4, 0.4), pos
            )  # Increased range
            pos[o] = (x, y)

    # Position remaining nodes
    remaining_nodes = set(G.nodes()) - set(pos.keys())
    for node in remaining_nodes:
        if node in subjects:
            base_x, base_y = -2, sentence_base_y + rng.uniform(-spacing, spacing)
        elif node in verbs:
            base_x, base_y = 0, sentence_base_y + rng.uniform(-spacing, spacing)
        else:
            base_x, base_y = 2, sentence_base_y + rng.uniform(-spacing, spacing)
        x, y = get_available_position(base_x, base_y, pos)
        pos[node] = (x, y)

    # Collect weights of syntactic edges
    syntactic_weights = [
        data["weight"]
        for u, v, data in G.edges(data=True)
        if "syntactic" in data.get("relation", set())
    ]
    if syntactic_weights:
        min_weight = min(syntactic_weights)
        max_weight = max(syntactic_weights)
    else:
        min_weight = max_weight = 1

    # Function to draw curved edges
    def draw_edge(u, v, rad, style, color, linewidth, alpha=0.3):
        arrow = FancyArrowPatch(
            posA=pos[u],
            posB=pos[v],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            linestyle=style,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        plt.gca().add_patch(arrow)

    # Draw edges
    for u, v, data in G.edges(data=True):
        # Determine edge style based on node types
        start_type = G.nodes[u].get("type", set())
        end_type = G.nodes[v].get("type", set())

        if ("subject" in start_type and "subject" in end_type) or (
            "object" in start_type and "object" in end_type
        ):
            style = "--"
        else:
            style = "-"

        # Mark passive_approx edges (only when explicitly requested and only
        # when EVERY contributing row was passive_approx=1, i.e. the edge
        # exists exclusively as a victim-in-Agent approximation).
        edge_weight = data.get("weight", 1)
        pa_count = data.get("passive_approx_count", 0)
        edge_is_passive_approx = (
            mark_passive_approx and edge_weight > 0 and pa_count >= edge_weight
        )
        if edge_is_passive_approx:
            style = ":"  # dotted
            extra_alpha_factor = 0.55
        else:
            extra_alpha_factor = 1.0

        # Determine edge color based on valence of start and end nodes
        start_valence = valences[u]
        end_valence = valences[v]

        relations = data.get("relation", set())
        if isinstance(relations, str):
            relations = set([relations])

        # Now handle the cases
        if "syntactic" in relations and "synonym" in relations:
            # Both relations exist

            # Determine weight-based linewidth for syntactic edge
            weight = data.get("weight", 1)
            if min_weight == max_weight:
                linewidth = 4  # Default linewidth if all weights are the same
            else:
                # Map weight to linewidth between 4.5 and 6.5
                linewidth = 4.5 + 2 * (weight - min_weight) / (max_weight - min_weight)

            # Determine edge color based on valence
            if start_valence == "positive" and end_valence == "positive":
                color = "#1f77b4"  # Blue
            elif start_valence == "negative" and end_valence == "negative":
                color = "#d62728"  # Red
            elif (start_valence == "positive" and end_valence == "negative") or (
                start_valence == "negative" and end_valence == "positive"
            ):
                color = "#9467bd"  # Purple
            elif (start_valence == "positive" and end_valence == "neutral") or (
                end_valence == "positive" and start_valence == "neutral"
            ):
                color = "#b4cad6"  # Grayish blue
            elif (start_valence == "negative" and end_valence == "neutral") or (
                end_valence == "negative" and start_valence == "neutral"
            ):
                color = "#dc9f9e"  # Grayish red
            else:
                color = "#7f7f7f"  # Grey

            # Draw syntactic edge with rad=0.1
            draw_edge(
                u,
                v,
                rad=0.08,
                style=style,
                color=color,
                linewidth=linewidth,
                alpha=0.45 * extra_alpha_factor,
            )

            # Then, draw synonym edge
            linewidth = 1.5  # Semantic relations have fixed linewidth of 2
            color = "#009E73"  # Green
            # For synonym edge, use rad=0.2 (more curved)
            draw_edge(
                u,
                v,
                rad=0.25,
                style=style,
                color=color,
                linewidth=linewidth,
                alpha=0.25 * extra_alpha_factor,
            )

        elif "syntactic" in relations:
            # Only syntactic relation

            # Determine weight-based linewidth
            weight = data.get("weight", 1)
            if min_weight == max_weight:
                linewidth = 4  # Default linewidth if all weights are the same
            else:
                # Map weight to linewidth between 4.5 and 6.5
                linewidth = 4.5 + 2 * (weight - min_weight) / (max_weight - min_weight)

            # Determine edge color based on valence
            if start_valence == "positive" and end_valence == "positive":
                color = "#1f77b4"  # Blue
            elif start_valence == "negative" and end_valence == "negative":
                color = "#d62728"  # Red
            elif (start_valence == "positive" and end_valence == "negative") or (
                start_valence == "negative" and end_valence == "positive"
            ):
                color = "#9467bd"  # Purple
            elif (start_valence == "positive" and end_valence == "neutral") or (
                end_valence == "positive" and start_valence == "neutral"
            ):
                color = "#a2b9d8"  # Grayish blue
            elif (start_valence == "negative" and end_valence == "neutral") or (
                end_valence == "negative" and start_valence == "neutral"
            ):
                color = "#dc9f9e"  # Grayish red
            else:
                color = "#7f7f7f"  # Grey

            # Draw syntactic edge with rad=0.1
            draw_edge(
                u,
                v,
                rad=0.08,
                style=style,
                color=color,
                linewidth=linewidth,
                alpha=0.45 * extra_alpha_factor,
            )

        elif "synonym" in relations:
            # Only synonym relation

            linewidth = 2  # Semantic relations have fixed linewidth of 2
            color = "#009E73"  # Green

            # Draw synonym edge with rad=0.15
            draw_edge(
                u,
                v,
                rad=0.14,
                style=style,
                color=color,
                linewidth=linewidth,
                alpha=0.35 * extra_alpha_factor,
            )

        else:
            # No valid relation to plot
            continue  # Skip this edge

    # Draw labels with rectangular backgrounds
    for node in G.nodes():
        x, y = pos[node]
        label = G.nodes[node].get("label", node)
        color = node_colors[node]

        bbox_props = dict(
            boxstyle="round,pad=0.45,rounding_size=0.3", fc=color, ec="none", alpha=0.85
        )
        plt.text(
            x,
            y,
            label,
            color="white",
            horizontalalignment="center",
            verticalalignment="center",
            bbox=bbox_props,
            fontsize=custom_font,
        )

    # Add column labels
    plt.text(
        -2,
        max(pos.values(), key=lambda x: x[1])[1] + 2.0,
        "AGENT",
        fontsize=40,
        ha="center",
    )
    plt.text(
        0,
        max(pos.values(), key=lambda x: x[1])[1] + 2.0,
        "EVENT",
        fontsize=40,
        ha="center",
    )
    plt.text(
        2,
        max(pos.values(), key=lambda x: x[1])[1] + 2.0,
        "TARGET",
        fontsize=40,
        ha="center",
    )

    # Adjust the y-axis limits accordingly
    y_max = max(pos.values(), key=lambda x: x[1])[1]
    y_min = min(pos.values(), key=lambda x: x[1])[1]
    plt.ylim(y_min - 3, y_max + 1.5)
    x_min = min(pos.values(), key=lambda x: x[0])[0]
    x_max = max(pos.values(), key=lambda x: x[0])[0]
    plt.xlim(x_min - 0.1, x_max + 0.1)

    plt.axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

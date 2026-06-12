import re

import networkx as nx
from .nlp_utils import compute_valence
from collections import defaultdict
import pandas as pd


################################################################################################
## SVO Operations.
################################################################################################


def filter_svo_dataframe_by_tea(df, TEA, TEA2=None):
    """
    Filters the SVO (Subject-Verb-Object) DataFrame to include only rows where 'TEA' and 'TEA2' match the provided arguments.
    This function specifically considers 'TEA' to be one of "Agent", "Event", or "Target". Synonyms of these terms are not taken into account.
    If `TEA2` is set to None , we consider all edges of TEA.

    Parameters:
    df (pd.DataFrame): The SVO DataFrame to filter.
    TEA: The value to match in the 'TEA' column. Accepted values are:
        - "Agent"
        - "Event"
        - "Target"
    TEA2: The value to match in the 'TEA2' column.
        - If `TEA2` is set to None , we consider all edges of TEA.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    # Define valid TEA values in lowercase for case-insensitive comparison
    valid_tea_values = {"target", "event", "agent"}

    # Normalize inputs to lowercase
    if not isinstance(TEA, str):
        raise ValueError(f"TEA must be a string. Provided type: {type(TEA)}")
    TEA_normalized = TEA.lower()

    if TEA_normalized not in valid_tea_values:
        raise ValueError(f"TEA must be one of {valid_tea_values}. Provided: '{TEA}'")

    if TEA2 is not None:
        if not isinstance(TEA2, str):
            raise ValueError(
                f"TEA2 must be a string or None. Provided type: {type(TEA2)}"
            )
        TEA2_normalized = TEA2.lower()
        if TEA2_normalized not in valid_tea_values:
            raise ValueError(
                f"TEA2 must be one of {valid_tea_values} or None. Provided: '{TEA2}'"
            )
    else:
        TEA2_normalized = None

    # Apply filtering
    if TEA2_normalized is None:
        filtered_df = df[
            (
                (df["TEA"].str.lower() == TEA_normalized)
                | (df["TEA2"].str.lower() == TEA_normalized)
            )
            & (df["Semantic-Syntactic"] == 0)
        ]
    else:
        filtered_df = df[
            (df["TEA"].str.lower() == TEA_normalized)
            & (df["TEA2"].str.lower() == TEA2_normalized)
            & (df["Semantic-Syntactic"] == 0)
        ]

    return filtered_df


def merge_svo_dataframes(df_list):
    """
    Merges a list of DataFrames containing SVO data into a single DataFrame.
    The 'svo_id' column is updated to ensure unique values across the merged DataFrame.

    Args:
        df_list (list): A list of DataFrames containing SVO data

    Returns:
        pandas.DataFrame: A single DataFrame containing all SVO data
    """
    merged_df = pd.DataFrame()
    svo_id_offset = 0
    for df in df_list:
        df_copy = df.copy()
        # Ensure svo_id is numeric, keeping NaNs
        df_copy["svo_id"] = pd.to_numeric(df_copy["svo_id"], errors="coerce")
        # Convert to nullable integer type to allow NaNs
        df_copy["svo_id"] = df_copy["svo_id"].astype("Int64")
        # Increment IDs only for non-null values
        df_copy.loc[df_copy["svo_id"].notna(), "svo_id"] += svo_id_offset
        merged_df = pd.concat([merged_df, df_copy], ignore_index=True)
        # Update offset for the next DataFrame, ignoring NaNs
        max_id = df_copy["svo_id"].max()
        if pd.notna(max_id):
            svo_id_offset = max_id + 1
    # Backward compatibility: if any of the input DataFrames was produced
    # before the ``passive_approx`` flag was introduced, the column will be
    # NaN for those rows after concat. Fill with 0 (= "no approximation
    # information available") to preserve a coherent integer dtype downstream.
    if "passive_approx" in merged_df.columns:
        merged_df["passive_approx"] = (
            merged_df["passive_approx"].fillna(0).astype(int)
        )
    if "is_passive" in merged_df.columns:
        merged_df["is_passive"] = (
            merged_df["is_passive"].fillna(0).astype(int)
        )
    return merged_df


def drop_approximated_svos(df):
    """
    Remove ALL rows belonging to SVO triples flagged as approximated passives
    (``passive_approx`` = 1 on their Agent-Event/Agent-Agent rows).

    Filtering row-by-row (``df[df["passive_approx"] == 0]``) is NOT enough:
    the Event-Target and Target-Target rows of an approximated triple always
    carry ``passive_approx`` = 0, so they would survive the filter. This
    helper drops the whole triple via its ``svo_id``.

    Args:
        df (pd.DataFrame): An SVO DataFrame produced by ``extract_svos``.

    Returns:
        pd.DataFrame: The DataFrame without any row of approximated triples.
    """
    if "passive_approx" not in df.columns:
        return df

    flags = pd.to_numeric(df["passive_approx"], errors="coerce").fillna(0).astype(int)
    approx_ids = set(df.loc[flags == 1, "svo_id"].unique())
    # Semantic rows have svo_id == "N/A" and no voice information: keep them.
    approx_ids.discard("N/A")
    return df[~df["svo_id"].isin(approx_ids)]


def export_hypergraphs(df):
    """
    Extracts hypergraphs from the DataFrame where Semantic-Syntactic is 0,
    allowing duplicates across sentences but not within the same sentence.

    Args:
        df (pandas.DataFrame): DataFrame containing the hypergraph data

    Returns:
        list: A list of hypergraph strings, potentially with duplicates
    """
    # Assume we have a 'sentence_id' column in the DataFrame
    # If not, you'll need to add logic to identify sentence boundaries

    hypergraphs = []
    sentence_hypergraphs = defaultdict(set)

    # Filter rows where Semantic-Syntactic is 0
    filtered_df = df[df["Semantic-Syntactic"] == 0]

    # Group by sentence_id and collect hypergraphs
    for svo_id, group in filtered_df.groupby("svo_id"):
        for hypergraph in group["Hypergraph"]:
            if hypergraph != "N/A" and hypergraph not in sentence_hypergraphs[svo_id]:
                hypergraphs.append(hypergraph)
                sentence_hypergraphs[svo_id].add(hypergraph)

    return hypergraphs


def export_subj(df):
    """
    Extracts a set of all unique subjects from the DataFrame.
    Each element in the set is a tuple of (subject, valence).
    """
    # Filter rows where TEA is 'Agent' (subjects sit in 'Node 1')
    df_subjects = df[df["TEA"] == "Agent"]
    # Get unique subjects from 'Node 1'
    subjects = df_subjects["Node 1"].unique()
    # Compute valence and create a set of tuples
    subject_tuples = set()
    for subj in subjects:
        valence = compute_valence(subj)
        subject_tuples.add((subj, valence))
    return subject_tuples


def export_obj(df):
    """
    Extracts a set of all unique objects from the DataFrame.
    Each element in the set is a tuple of (object, valence).
    """
    # Filter rows where TEA2 is 'Target' (objects sit in 'Node 2')
    df_objects = df[df["TEA2"] == "Target"]
    # Get unique objects from 'Node 2'
    objects = df_objects["Node 2"].unique()
    # Compute valence and create a set of tuples
    object_tuples = set()
    for obj in objects:
        valence = compute_valence(obj)
        object_tuples.add((obj, valence))
    return object_tuples


def export_verb(df):
    """
    Extracts a set of all unique verbs from the DataFrame.
    Each element in the set is a tuple of (verb, valence).
    """
    # Verbs can be in 'Node 1' where TEA is 'Event' or in 'Node 2' where TEA2 is 'Event'
    verbs_node1 = df[df["TEA"] == "Event"]["Node 1"]
    verbs_node2 = df[df["TEA2"] == "Event"]["Node 2"]
    # Combine and get unique verbs
    verbs = pd.concat([verbs_node1, verbs_node2]).unique()
    # Compute valence and create a set of tuples
    verb_tuples = set()
    for verb in verbs:
        valence = compute_valence(verb)
        verb_tuples.add((verb, valence))
    return verb_tuples


def add_node_with_type(G, node_id, label, node_type):
    """
    Add a node to the graph with a specific type.
    If the node already exists, update its type to include the new type.
    """
    if G.has_node(node_id):
        if "type" in G.nodes[node_id]:
            G.nodes[node_id]["type"].add(node_type)
        else:
            G.nodes[node_id]["type"] = set([node_type])
    else:
        G.add_node(node_id, type=set([node_type]), label=label)


def _term_pattern(term):
    """
    Build a safe, whole-word, case-insensitive regex pattern for *term*.
    Escaping prevents regex injection (e.g. terms containing '(' or '$');
    word boundaries prevent substring false positives (e.g. 'he' matching
    'the teacher').
    """
    return rf"\b{re.escape(term)}\b"


def filter_subjects(df, subject_term):
    """
    Filters the DataFrame to keep rows whose 'svo_id' corresponds to entries
    where 'Node 1' contains the subject term as a whole word and 'TEA' is
    'Agent'.

    Parameters:
    - df: The input DataFrame.
    - subject_term: The term to search for in subjects (matched case-insensitively, whole word).

    Returns:
    - A filtered DataFrame containing only rows with matching 'svo_id's.
    """
    # Identify 'svo_id's where 'Node 1' contains the subject term and 'TEA' is 'Agent'
    subject_svo_ids = set(
        df[
            (df["Node 1"].str.contains(_term_pattern(subject_term), case=False, na=False, regex=True))
            & (df["TEA"] == "Agent")
        ]["svo_id"].unique()
    )

    # Filter the DataFrame to only include rows with these 'svo_id's
    filtered_df = df[df["svo_id"].isin(subject_svo_ids)]
    return filtered_df


def filter_objects(df, object_term):
    """
    Filters the DataFrame to keep rows whose 'svo_id' corresponds to entries
    where the object term appears as a whole word in a 'Target' node.

    Object nodes live in 'Node 2' of Event-Target rows (TEA2 == 'Target') and
    in both columns of Target-Target rows (TEA == 'Target').

    Parameters:
    - df: The input DataFrame.
    - object_term: The term to search for in objects (matched case-insensitively, whole word).

    Returns:
    - A filtered DataFrame containing only rows with matching 'svo_id's.
    """
    pattern = _term_pattern(object_term)
    in_node2 = (
        df["Node 2"].str.contains(pattern, case=False, na=False, regex=True)
        & (df["TEA2"] == "Target")
    )
    in_node1 = (
        df["Node 1"].str.contains(pattern, case=False, na=False, regex=True)
        & (df["TEA"] == "Target")
    )
    object_svo_ids = set(df[in_node2 | in_node1]["svo_id"].unique())
    # Filter the DataFrame to only include rows with these 'svo_id's
    filtered_df = df[df["svo_id"].isin(object_svo_ids)]
    return filtered_df


def svo_to_graph(df, subject_filter=None, object_filter=None):
    """
    Convert a pandas DataFrame of SVO data into a graph.
    Optionally filters the data based on the subject_filter.
    """
    G = nx.Graph()

    if subject_filter is not None:
        df = filter_subjects(df, subject_filter)

    if object_filter is not None:
        df = filter_objects(df, object_filter)

    for index, row in df.iterrows():
        node1 = row["Node 1"]
        wdw1 = row["TEA"]
        node2 = row["Node 2"]
        tea2 = row["TEA2"]

        # Skip rows with missing node labels to prevent TypeError
        if pd.isna(node1) or pd.isna(node2):
            continue
        hypergraph = row["Hypergraph"]
        sem_synt = row["Semantic-Syntactic"]
        # Backward compatible: legacy DataFrames produced before the
        # passive_approx flag was introduced will not have the column at all,
        # and merged legacy+new DataFrames may contain NaN. Both cases default
        # to 0 (i.e. "no approximation information available").
        if "passive_approx" in df.columns:
            _pa = row["passive_approx"]
            passive_approx = 0 if pd.isna(_pa) else int(_pa)
        else:
            passive_approx = 0

        # Determine node types based on TEA and TEA2
        node1_type = (
            "subject" if wdw1 == "Agent" else "verb" if wdw1 == "Event" else "object"
        )
        node2_type = (
            "subject" if tea2 == "Agent" else "verb" if tea2 == "Event" else "object"
        )

        # Create unique node IDs to differentiate between subjects, verbs, and objects
        node1_id = node1 + "_" + node1_type[0]
        node2_id = node2 + "_" + node2_type[0]

        # Add nodes with labels and types
        add_node_with_type(G, node1_id, label=node1, node_type=node1_type)
        add_node_with_type(G, node2_id, label=node2, node_type=node2_type)

        # Determine relation type based on 'Semantic-Syntactic' column
        relation_type = "synonym" if sem_synt == 1 else "syntactic"

        if G.has_edge(node1_id, node2_id):
            # Edge exists, update the 'relation' attribute
            existing_relation = G.edges[node1_id, node2_id].get("relation", set())
            if isinstance(existing_relation, str):
                existing_relation = set([existing_relation])
            existing_relation.add(relation_type)
            G.edges[node1_id, node2_id]["relation"] = existing_relation
            # Also update hypergraph
            existing_hypergraph = G.edges[node1_id, node2_id].get("hypergraph", set())
            if isinstance(existing_hypergraph, str):
                existing_hypergraph = set([existing_hypergraph])
            existing_hypergraph.add(hypergraph)
            G.edges[node1_id, node2_id]["hypergraph"] = existing_hypergraph
            # Increment weight for syntactic relations
            if relation_type == "syntactic":
                G.edges[node1_id, node2_id]["weight"] += 1
            # Track passive_approx occurrences (count of victim-in-Agent edges
            # collapsed onto this graph edge). An edge is considered fully
            # "approximated" only if every contributing row was passive_approx=1.
            G.edges[node1_id, node2_id]["passive_approx_count"] = (
                G.edges[node1_id, node2_id].get("passive_approx_count", 0)
                + passive_approx
            )
        else:
            # Initialize weight to 1 for syntactic relations
            weight = 1 if relation_type == "syntactic" else 0
            G.add_edge(
                node1_id,
                node2_id,
                relation=set([relation_type]),
                hypergraph=set([hypergraph]),
                weight=weight,
                passive_approx_count=passive_approx,
            )

    return G


################################################################################################
## Centrality measures.
################################################################################################


def tea_weighted_degree_centrality(
    df,
    TEA,
    TEA2=None,
    remove_same_type=False,
    remove_node_type=None,
):
    """
    Filters the SVO (Subject-Verb-Object) DataFrame to include only rows where 'TEA' and 'TEA2' match the provided arguments.
    This function specifically considers 'TEA' to be one of "Agent", "Event", or "Target". Synonyms of these terms are not taken into account.
    If `TEA2` is set to None , we consider all edges of TEA.

    Parameters:
    df (pd.DataFrame): The SVO DataFrame to filter.
    TEA: The value to match in the 'TEA' column. Accepted values are:
        - "Agent"
        - "Event"
        - "Target"
    TEA2: The value to match in the 'TEA2' column.
        - If `TEA2` is set to None , we consider all edges of TEA.

    remove_same_type (bool): If True, removes edges between nodes of the same type (target-target, agent-agent).
    remove_node_type (str): If set to "Target" or "Agent", removes all rows containing that node type.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    # First filter the DataFrame based on node_type parameter
    if remove_node_type in ["Target", "Agent"]:
        df = df[
            ~df["TEA"].isin([remove_node_type]) & ~df["TEA2"].isin([remove_node_type])
        ]

    filtered_df = filter_svo_dataframe_by_tea(df, TEA, TEA2)

    if remove_same_type:
        # Keep only rows where TEA and TEA2 are different
        filtered_df = filtered_df[filtered_df["TEA"] != filtered_df["TEA2"]]

    G = svo_to_graph(filtered_df)

    # Compute weighted degrees (node strengths)
    node_strength = dict(G.degree(weight="weight"))
    # Normalize by total strength to get centrality
    total_strength = sum(node_strength.values())
    weighted_degree_centrality = {
        node: strength / total_strength for node, strength in node_strength.items()
    }
    # Create a DataFrame with 'node' as a column
    df = pd.DataFrame(
        list(weighted_degree_centrality.items()), columns=["node", "degree_centrality"]
    )

    # Filter the DataFrame to include only subjects, verbs, or objects
    if TEA == "Agent":
        df_filtered = df[df["node"].str.endswith("_s")].copy()
    elif TEA == "Event" and TEA2 is None:
        df_filtered = df[df["node"].str.endswith("_v")].copy()
    elif TEA2 == "Target":
        df_filtered = df[df["node"].str.endswith("_o")].copy()
    else:
        raise ValueError(
            f"Unsupported (TEA, TEA2) combination: ({TEA!r}, {TEA2!r}). "
            "Supported combinations are: ('Agent', ...) for subjects, "
            "('Event', None) for verbs, (..., 'Target') for objects."
        )
    # Remove the last character(s) from the node names
    df_filtered["node"] = df_filtered["node"].str.replace("(_s|_v|_o)$", "", regex=True)

    # Sort the DataFrame
    df_sorted = df_filtered.sort_values(
        by="degree_centrality", ascending=False
    ).reset_index(drop=True)
    return df_sorted


def tea_degree_centrality_overview(df):
    """
    Compute the degree centrality for each of the SVO components (Subject, Verb, Object) using the weighted degree centrality measure.

    Returns:
    dict[str, pd.DataFrame]: The full centrality DataFrames, keyed by
    "Subject", "Verb", and "Object" (the top 20 rows of each are also
    printed/displayed).
    """
    try:
        from IPython.display import display as _display
    except ImportError:
        _display = print

    combinations = [
        ["Subject", "Agent", "Event"],
        ["Verb", "Event", None],
        ["Object", "Event", "Target"],
    ]

    results = {}
    for svo, TEA, TEA2 in combinations:
        centrality_df = tea_weighted_degree_centrality(df, TEA, TEA2)
        results[svo] = centrality_df
        print(f"Degree centrality for {svo}")
        _display(centrality_df.head(20))
        print("############################################ \n")

    return results

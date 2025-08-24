# -*- coding: utf-8 -*-
"""
Obesity research in specialty journals from 2000 to 2023: A bibliometric analysis
"""

# ==========================================
# 1. LIBRARY INSTALLATION AND IMPORTS
# ==========================================
!pip install python-igraph networkx matplotlib seaborn pandas numpy plotly

import os
from google.colab import drive
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import ast
import logging
import warnings
from itertools import combinations
import plotly.graph_objects as go
import plotly.offline as pyo
from matplotlib.colors import ListedColormap
import random

# Configure plotting and logging
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Mount Google Drive
drive.mount('/content/drive')

# Define the path
output_dir = '/content/drive/My Drive/DATASETS/OBESITY.JOURNALS/'

# Load dataset
file_name = os.path.join(output_dir, 'merged_results_filtered.csv')
df = pd.read_csv(file_name)
df.head()

df.info()

# NETWORK ANALYSIS
# ==========================================
# Co-authorship Network Analysis with Temporal Evolution
#
# Scientific Background:
# - Co-authorship networks map collaborative structures in science (Newman, 2001)
# - Temporal analysis reveals evolution of collaboration patterns (Glänzel, 2003)
# - Centrality measures capture different dimensions of scientific influence
# - Community detection identifies research clusters (Pons & Latapy, 2005)
# - Isolated nodes are excluded from visualization as they don't represent collaboration


# ==========================================
# 2. TEMPORAL PERIODS DEFINITION
# ==========================================

# Define analysis periods for temporal evolution study
# Based on bibliometric best practices for longitudinal analysis (Börner et al., 2004)
periods = {
    '2000-2007': (2000, 2007),  # Early 2000s - Foundation period
    '2008-2015': (2008, 2015),  # 2008-2015 - Growth period
    '2016-2023': (2016, 2023)  # 2016-2023 - Recent period
    #'2000-2023': (2000, 2023)   # Complete period - Overall analysis
}

# ==========================================
# 3. DATA PREPROCESSING AND VALIDATION
# ==========================================

def validate_researchers_data(df):
    """
    Validate the researchers column for network construction.

    Scientific Rationale:
    Data quality is crucial for reliable network analysis (Newman, 2001).
    Invalid or missing author data can create spurious network patterns.
    """
    print(" Validating researchers data quality...")

    # Check for missing researchers data
    null_researchers = df['researchers'].isnull().sum()
    empty_researchers = (df['researchers'].apply(lambda x: len(x) if isinstance(x, list) else 0) == 0).sum()

    print(f"   - Null researchers: {null_researchers:,}")
    print(f"   - Empty researchers: {empty_researchers:,}")
    print(f"   - Valid records: {len(df) - null_researchers - empty_researchers:,}")

    # Sample researchers structure
    sample_researchers = df[df['researchers'].notna()]['researchers'].iloc[0]
    print(f"   - Sample structure: {type(sample_researchers)}")

    return True

# Validate input data
validate_researchers_data(df)

def extract_author_collaborations(df_period):
    """
    Extract author collaboration pairs from bibliographic data.

    Scientific Method:
    - Creates undirected edges between all co-authors on each paper
    - Weights edges by collaboration frequency (Newman, 2001)
    - Uses author IDs for unique identification across publications

    Args:
        df_period: DataFrame filtered for specific time period

    Returns:
        collaborations: dict with (author1, author2) keys and frequency values
        author_info: dict with author metadata
    """
    collaborations = defaultdict(int)
    author_info = {}
    total_papers = 0
    collaboration_papers = 0

    for idx, row in df_period.iterrows():
        try:
            researchers = row.get('researchers', [])

            # Handle string representation of lists
            if isinstance(researchers, str):
                try:
                    researchers = ast.literal_eval(researchers)
                except:
                    continue

            if not isinstance(researchers, list) or len(researchers) < 2:
                continue

            total_papers += 1

            # Extract author IDs and metadata
            author_ids = []
            for researcher in researchers:
                if isinstance(researcher, dict) and 'id' in researcher:
                    author_id = researcher['id']
                    author_ids.append(author_id)

                    # Store author metadata for visualization
                    if author_id not in author_info:
                        first_name = researcher.get('first_name', '')
                        last_name = researcher.get('last_name', '')
                        author_info[author_id] = {
                            'name': f"{first_name} {last_name}".strip() or author_id,
                            'first_name': first_name,
                            'last_name': last_name,
                            'papers': 0,
                            'research_orgs': researcher.get('research_orgs', [])
                        }
                    author_info[author_id]['papers'] += 1

            # Create collaboration edges (all pairs of co-authors)
            if len(author_ids) >= 2:
                collaboration_papers += 1
                for author1, author2 in combinations(author_ids, 2):
                    # Ensure consistent edge direction (smaller ID first)
                    edge = tuple(sorted([author1, author2]))
                    collaborations[edge] += 1

        except Exception as e:
            logger.error(f"Error processing paper {row.get('id', 'unknown')}: {str(e)}")
            continue

    print(f"Extracted {len(collaborations):,} unique collaborations")
    print(f"From {collaboration_papers:,} collaborative papers ({total_papers:,} total)")
    print(f"Involving {len(author_info):,} unique authors")

    return dict(collaborations), author_info

# ==========================================
# 4. NETWORK CONSTRUCTION AND ANALYSIS
# ==========================================

def build_coauthorship_network(collaborations, author_info):
    """
    Construct co-authorship network using NetworkX.

    Scientific Method:
    - Undirected weighted graph where nodes = authors, edges = collaborations
    - Edge weights = collaboration frequency (Newman, 2001)
    - Node attributes include author metadata for analysis

    Returns:
        NetworkX Graph object with author collaboration network
    """
    print("  Building co-authorship network...")

    # Create undirected weighted graph
    G = nx.Graph()

    # Add nodes with attributes
    for author_id, info in author_info.items():
        G.add_node(author_id,
                   name=info['name'],
                   papers=info['papers'],
                   research_orgs=info.get('research_orgs', []))

    # Add weighted edges
    for (author1, author2), weight in collaborations.items():
        G.add_edge(author1, author2, weight=weight)

    # Report network statistics
    isolated_nodes = list(nx.isolates(G))
    print(f"Network created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f" Isolated nodes: {len(isolated_nodes):,}")
    if G.number_of_nodes() > len(isolated_nodes):
        avg_degree = 2 * G.number_of_edges() / (G.number_of_nodes() - len(isolated_nodes))
        print(f"Average degree (connected nodes): {avg_degree:.2f}")

    return G

def remove_isolated_nodes(G, purpose="visualization"):
    """
    Remove isolated nodes from the network.

    Scientific Rationale:
    - Isolated nodes don't contribute to collaboration analysis
    - They create visual clutter in network visualizations
    - Community detection is not meaningful for single nodes
    - Centrality measures are zero or undefined for isolated nodes

    Args:
        G: NetworkX graph
        purpose: string describing why we're removing isolated nodes

    Returns:
        NetworkX graph without isolated nodes
    """
    isolated_nodes = list(nx.isolates(G))

    if isolated_nodes:
        print(f"Removing {len(isolated_nodes):,} isolated nodes for {purpose}")
        G_connected = G.copy()
        G_connected.remove_nodes_from(isolated_nodes)

        print(f"Connected network: {G_connected.number_of_nodes():,} nodes, {G_connected.number_of_edges():,} edges")
        return G_connected
    else:
        print(f"No isolated nodes found - network is fully connected")
        return G.copy()

def calculate_centrality_measures(G):
    """
    Calculate multiple centrality measures for author importance ranking.

    Scientific Rationale:
    - Degree Centrality: measures direct collaboration activity (Freeman, 1978)
    - Betweenness Centrality: identifies brokers connecting different groups (Freeman, 1977)
    - PageRank: measures prestige and influence in network (Page et al., 1999)
    - Different measures capture different aspects of scientific influence
    - Calculated only on connected nodes (isolated nodes would have zero/undefined values)

    Returns:
        Dictionary with centrality scores for each author
    """
    print(" Calculating centrality measures...")

    # Work with connected network only for meaningful centrality calculations
    G_connected = remove_isolated_nodes(G, "centrality calculation")

    centralities = {}

    if G_connected.number_of_nodes() > 0:
        # Degree Centrality - measures direct collaboration activity
        centralities['degree'] = nx.degree_centrality(G_connected)

        # Betweenness Centrality - identifies brokers and bridges
        centralities['betweenness'] = nx.betweenness_centrality(G_connected, normalized=True)

        # PageRank - measures prestige and influence
        centralities['pagerank'] = nx.pagerank(G_connected, alpha=0.85)

        # Closeness Centrality - measures how close an author is to all others
        centralities['closeness'] = nx.closeness_centrality(G_connected)

        print("    Centrality measures calculated:")
        for measure in centralities.keys():
            print(f"  - {measure.capitalize()}: ✓")
    else:
        print("     No connected nodes found - cannot calculate centralities")
        centralities = {'degree': {}, 'betweenness': {}, 'pagerank': {}, 'closeness': {}}

    return centralities

def select_top_authors(G, centralities, top_n=50):
    """
    Select top N most influential authors based on aggregated centrality measures.

    Scientific Method:
    - Combines multiple centrality measures using z-score normalization
    - Prevents bias from single centrality measure (Borgatti, 2005)
    - Enables focus on most influential collaboration subnetwork
    - Only considers connected authors (isolated nodes excluded)

    Args:
        G: NetworkX graph
        centralities: dict of centrality measures
        top_n: number of top authors to select

    Returns:
        List of top author IDs and their aggregated scores
    """
    print(f" Selecting top {top_n} most influential authors...")

    # Work with connected nodes only
    connected_nodes = [node for node in G.nodes() if G.degree(node) > 0]

    if not connected_nodes:
        print("     No connected nodes found")
        return [], {}

    # Normalize centrality measures using z-scores
    normalized_centralities = {}
    for measure, scores in centralities.items():
        if scores:  # Only process if we have scores
            values = np.array([scores.get(node, 0) for node in connected_nodes])
            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val > 0:
                normalized_centralities[measure] = {
                    node: (scores.get(node, 0) - mean_val) / std_val
                    for node in connected_nodes
                }
            else:
                normalized_centralities[measure] = {node: 0 for node in connected_nodes}

    if not normalized_centralities:
        print("     No centrality measures available")
        return connected_nodes[:top_n], {node: 0 for node in connected_nodes[:top_n]}

    # Aggregate normalized centralities (equal weights)
    aggregated_scores = {}
    for node in connected_nodes:
        score = sum(normalized_centralities[measure].get(node, 0)
                   for measure in normalized_centralities.keys())
        aggregated_scores[node] = score

    # Select top N authors
    top_authors = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_author_ids = [author for author, score in top_authors]

    print(f"Top author selection complete")
    if top_authors:
        print(f"Score range: {top_authors[-1][1]:.3f} to {top_authors[0][1]:.3f}")

    return top_author_ids, dict(top_authors)

def create_subnetwork(G, top_authors):
    """
    Create subnetwork containing only top influential authors.

    Purpose: Focus visualization and community detection on most important actors
    Note: This will automatically exclude isolated nodes as they won't be in top authors
    """
    subG = G.subgraph(top_authors).copy()

    # Double-check for isolated nodes in subnetwork (shouldn't happen but good practice)
    subG_connected = remove_isolated_nodes(subG, "subnetwork creation")

    print(f"Subnetwork created: {subG_connected.number_of_nodes():,} nodes, {subG_connected.number_of_edges():,} edges")

    return subG_connected

# ==========================================
# 5. COMMUNITY DETECTION USING WALKTRAP
# ==========================================

def detect_communities_walktrap(G):
    """
    Apply Walktrap community detection algorithm using python-igraph.

    Scientific Method:
    Walktrap algorithm (Pons & Latapy, 2005):
    - Based on random walks on the network
    - Idea: random walks tend to stay within communities
    - Hierarchical clustering of random walk distances
    - Particularly effective for scientific collaboration networks
    - Only meaningful on connected networks (isolated nodes excluded)

    Args:
        G: NetworkX graph (should be connected, no isolated nodes)

    Returns:
        Community assignment for each node
    """
    print(" Applying Walktrap community detection...")

    # Ensure we're working with connected network
    G_connected = remove_isolated_nodes(G, "community detection")

    if G_connected.number_of_nodes() == 0:
        print("     No connected nodes for community detection")
        return {}, 0.0

    if G_connected.number_of_edges() == 0:
        print("     No edges for community detection")
        # Return each node as its own community
        return {node: i for i, node in enumerate(G_connected.nodes())}, 0.0

    # Convert NetworkX graph to igraph for Walktrap algorithm
    edge_list = [(u, v, G_connected[u][v]['weight']) for u, v in G_connected.edges()]
    node_list = list(G_connected.nodes())

    # Create igraph object
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_list))
    ig_graph.vs['name'] = node_list

    # Add edges with weights
    edges_to_add = []
    weights = []
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    for u, v, weight in edge_list:
        edges_to_add.append((node_to_idx[u], node_to_idx[v]))
        weights.append(weight)

    ig_graph.add_edges(edges_to_add)
    ig_graph.es['weight'] = weights

    # Apply Walktrap community detection
    # steps=4 is the default parameter (Pons & Latapy, 2005)
    walktrap = ig_graph.community_walktrap(weights='weight', steps=4)
    communities = walktrap.as_clustering()

    # Convert back to NetworkX node mapping
    community_assignment = {}
    for i, community_id in enumerate(communities.membership):
        node_name = node_list[i]
        community_assignment[node_name] = community_id

    print(f" Found {len(set(communities.membership))} communities")
    print(f"Modularity: {communities.modularity:.3f}")

    # Community size distribution
    community_sizes = Counter(communities.membership)
    print(f"Community sizes: {dict(community_sizes)}")

    return community_assignment, communities.modularity

# ==================================================
# 6. NETWORK VISUALIZATION (REMOVING ISOLATED NODES)
# ==================================================

def visualize_network(G, centralities, communities, period_name, top_authors_scores):
    """
    Create publication-quality network visualization with isolated nodes removed.

    Scientific Visualization Method:
    - Removes isolated nodes before visualization (collaboration networks focus)
    - Fruchterman-Reingold layout: force-directed, minimizes edge crossings (1991)
    - Node size proportional to PageRank centrality (influence)
    - Edge thickness proportional to collaboration frequency
    - Community colors for distinct group identification
    - High-resolution output suitable for publication

    Args:
        G: NetworkX graph (may contain isolated nodes)
        centralities: centrality measures
        communities: community assignments
        period_name: time period for title
        top_authors_scores: aggregated influence scores
    """
    print(f" Creating visualization for {period_name}...")

    # Remove isolated nodes for visualization
    G_viz = remove_isolated_nodes(G, "visualization")

    if G_viz.number_of_nodes() == 0:
        print(f" No connected nodes to visualize for {period_name}")
        return None

    # Set up the figure with publication quality
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=300)

    # Calculate Fruchterman-Reingold layout
    # Scientific justification: produces interpretable spatial arrangement (Fruchterman & Reingold, 1991)
    pos = nx.spring_layout(G_viz, k=3, iterations=50, seed=42)

    # Prepare node attributes
    node_sizes = []
    node_colors = []
    unique_communities = sorted(set(communities.values()))
    community_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
    community_color_map = {comm: color for comm, color in enumerate(community_colors)}

    for node in G_viz.nodes():
        # Node size based on PageRank centrality (scientific influence)
        pagerank_score = centralities['pagerank'].get(node, 0)
        size = 300 + (pagerank_score * 3000)  # Scale for visibility
        node_sizes.append(size)

        # Node color based on community membership
        community = communities.get(node, 0)
        node_colors.append(community_color_map[community])

    # Prepare edge attributes
    edge_weights = [G_viz[u][v]['weight'] for u, v in G_viz.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 * (weight / max_weight) + 0.5 for weight in edge_weights]

    # Draw network
    nx.draw_networkx_edges(G_viz, pos,
                          width=edge_widths,
                          alpha=0.6,
                          edge_color='gray',
                          ax=ax)

    nx.draw_networkx_nodes(G_viz, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.8,
                          linewidths=1,
                          edgecolors='black',
                          ax=ax)

    # Add labels for top 50 most influential authors only (avoid overcrowding)
    # Filter to only include connected nodes
    connected_top_authors = [(author, score) for author, score in top_authors_scores.items()
                           if author in G_viz.nodes()]
    top_50_authors = sorted(connected_top_authors, key=lambda x: x[1], reverse=True)[:50] # 50 or change this

    labels = {}
    for author, _ in top_50_authors: # 50 or change this
        # Change how author's name is displayed on the visualization:
        full_name = G_viz.nodes[author]['name']
        # split the name into parts
        parts = full_name.split()
        # get the last name (always the last word)
        last_name = parts[-1]
        # get initials from all previous parts
        initials = ' '.join([p[0] + '.' for p in parts[:-1]])
        # combine initials and last name
        name = f"{initials} {last_name}".strip()
        # if len(name) > 20:          # This is in case the name is too long
        #    name = name[:20] + '...'
        labels[author] = name

    nx.draw_networkx_labels(G_viz, pos, labels, font_size=8, font_weight='bold', ax=ax)

    # Customize plot
    isolated_count = G.number_of_nodes() - G_viz.number_of_nodes()
    title = (f'Co-authorship Network Analysis: {period_name}\n'
            f'Connected Nodes: {G_viz.number_of_nodes()}, Edges: {G_viz.number_of_edges()}, '
            f'Communities: {len(unique_communities)}')

    if isolated_count > 0:
        title += f'\n(Excluded {isolated_count} isolated nodes from visualization)'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add legend
    legend_elements = []
    for comm, color in community_color_map.items():
        comm_size = sum(1 for c in communities.values() if c == comm)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=10,
                                         label=f'Community {comm+1} (n={comm_size})'))

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Remove axes for cleaner look
    ax.axis('off')

    # Tight layout for publication
    plt.tight_layout()

    # Save high-resolution figure
    filename = f'coauthorship_network_{period_name.replace("-", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved: {filename}")

    plt.show()

    return fig

# ==========================================
# 7. CSV EXPORT FUNCTIONS
# ==========================================

def export_network_to_csv(G, centralities, communities, period_name, output_dir="network_exports"):
    """
    Export network data to CSV format for external analysis and publication tables.

    Note: Exports ALL nodes including isolated ones for complete data records,
    but marks them clearly for analysis purposes.

    Creates two CSV files per network:
    1. Nodes table: author information, centralities, communities
    2. Edges table: collaboration relationships and weights

    Args:
        G: NetworkX graph (original graph including isolated nodes)
        centralities: dict of centrality measures
        communities: community assignments
        period_name: time period identifier
        output_dir: directory for output files
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f" Exporting {period_name} network to CSV format...")

    # ========================================
    # NODES TABLE EXPORT (INCLUDING ISOLATED NODES)
    # ========================================
    nodes_data = []
    isolated_nodes = list(nx.isolates(G))

    for node in G.nodes():
        node_info = G.nodes[node]
        is_isolated = node in isolated_nodes

        # Basic node information
        row = {
            'author_id': node,
            'author_name': node_info.get('name', 'Unknown'),
            'papers_count': node_info.get('papers', 0),
            'is_isolated': is_isolated,
            'community': communities.get(node, -1) if not is_isolated else -1,
            'degree': G.degree(node),
            'weighted_degree': sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
        }

        # Add centrality measures (will be 0 or NaN for isolated nodes)
        for measure, scores in centralities.items():
            row[f'centrality_{measure}'] = scores.get(node, 0)

        # Add research organizations if available
        research_orgs = node_info.get('research_orgs', [])
        row['research_orgs'] = '; '.join(research_orgs) if research_orgs else ''
        row['num_research_orgs'] = len(research_orgs)

        nodes_data.append(row)

    # Create nodes DataFrame and sort by influence
    nodes_df = pd.DataFrame(nodes_data)
    nodes_df = nodes_df.sort_values('centrality_pagerank', ascending=False)

    # Export nodes table
    nodes_filename = os.path.join(output_dir, f'nodes_{period_name.replace("-", "_")}.csv')
    nodes_df.to_csv(nodes_filename, index=False)

    isolated_count = len(isolated_nodes)
    connected_count = len(nodes_df) - isolated_count
    print(f"Nodes exported: {nodes_filename}")
    print(f"  - Connected authors: {connected_count}")
    print(f"  - Isolated authors: {isolated_count}")
    print(f"  - Total authors: {len(nodes_df)}")

    # ========================================
    # EDGES TABLE EXPORT (ONLY CONNECTED NODES)
    # ========================================
    edges_data = []

    for u, v, data in G.edges(data=True):
        row = {
            'author1_id': u,
            'author2_id': v,
            'author1_name': G.nodes[u].get('name', 'Unknown'),
            'author2_name': G.nodes[v].get('name', 'Unknown'),
            'collaboration_weight': data.get('weight', 1),
            'author1_community': communities.get(u, -1),
            'author2_community': communities.get(v, -1),
            'same_community': communities.get(u, -1) == communities.get(v, -1),
            'author1_pagerank': centralities.get('pagerank', {}).get(u, 0),
            'author2_pagerank': centralities.get('pagerank', {}).get(v, 0)
        }

        edges_data.append(row)

    # Create edges DataFrame and sort by collaboration weight
    edges_df = pd.DataFrame(edges_data)
    edges_df = edges_df.sort_values('collaboration_weight', ascending=False)

    # Export edges table
    edges_filename = os.path.join(output_dir, f'edges_{period_name.replace("-", "_")}.csv')
    edges_df.to_csv(edges_filename, index=False)
    print(f"Edges exported: {edges_filename} ({len(edges_df)} collaborations)")

    return nodes_df, edges_df

def export_summary_statistics_csv(all_results, output_dir="network_exports"):
    """
    Export network summary statistics across all periods to CSV.

    Creates comprehensive summary table for publication including isolated node statistics.
    """

    os.makedirs(output_dir, exist_ok=True)

    print(" Exporting network summary statistics...")

    summary_data = []

    for period_name, results in all_results.items():
        if results:
            G = results['network']  # This is the subnetwork (top authors, connected)
            full_G = results['full_network']  # This includes all nodes
            centralities = results['centralities']
            communities = results['communities']

            # Calculate network metrics
            isolated_nodes_full = list(nx.isolates(full_G))
            isolated_nodes_sub = list(nx.isolates(G))

            row = {
                'period': period_name,
                'total_authors': full_G.number_of_nodes(),
                'isolated_authors_total': len(isolated_nodes_full),
                'connected_authors_total': full_G.number_of_nodes() - len(isolated_nodes_full),
                'top_authors_analyzed': G.number_of_nodes(),
                'isolated_authors_analyzed': len(isolated_nodes_sub),
                'total_collaborations': full_G.number_of_edges(),
                'analyzed_collaborations': G.number_of_edges(),
                'communities_count': len(set(communities.values())),
                'modularity': results['modularity'],
                'average_degree': 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
                'density': nx.density(G),
                'clustering_coefficient': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0,
                'max_pagerank': max(centralities['pagerank'].values()) if centralities['pagerank'] else 0,
                'max_betweenness': max(centralities['betweenness'].values()) if centralities['betweenness'] else 0,
                'max_degree_centrality': max(centralities['degree'].values()) if centralities['degree'] else 0
            }

            # Add connected components info
            components = list(nx.connected_components(G))
            row['connected_components'] = len(components)
            row['largest_component_size'] = len(max(components, key=len)) if components else 0

            summary_data.append(row)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Export summary table
    summary_filename = os.path.join(output_dir, 'network_summary_statistics.csv')
    summary_df.to_csv(summary_filename, index=False)
    print(f"Summary exported: {summary_filename}")

    return summary_df

# ===========================================
# 8. EXPORT NETWORK DATA (GraphML Compatible)
# ===========================================

def export_network_data(G, communities, centralities, period_name):
    """
    Export network data for external analysis and reproducibility.
    FIXED: Convert list attributes to strings for GraphML compatibility.

    Scientific Practice:
    - GraphML format preserves all network attributes
    - Enables analysis in other software (Gephi, Cytoscape)
    - Supports reproducibility and data sharing requirements
    - Exports connected network for meaningful analysis
    """
    print(f" Exporting network data for {period_name}...")

    # Work with connected network for meaningful export
    G_connected = remove_isolated_nodes(G, "external analysis export")

    # Create a copy to avoid modifying the original
    G_export = G_connected.copy()

    # Convert list attributes to strings for GraphML compatibility
    for node in G_export.nodes():
        # Convert research_orgs list to semicolon-separated string
        research_orgs = G_export.nodes[node].get('research_orgs', [])
        if isinstance(research_orgs, list):
            G_export.nodes[node]['research_orgs'] = '; '.join(research_orgs)

        # Ensure all other attributes are GraphML-compatible
        for attr_name, attr_value in G_export.nodes[node].items():
            if isinstance(attr_value, list):
                G_export.nodes[node][attr_name] = '; '.join(map(str, attr_value))

    # Add community assignments to nodes
    nx.set_node_attributes(G_export, communities, 'community')

    # Add centrality measures to nodes
    for measure, scores in centralities.items():
        filtered_scores = {node: scores.get(node, 0) for node in G_export.nodes()}
        nx.set_node_attributes(G_export, filtered_scores, f'centrality_{measure}')

    try:
        # Export as GraphML (preserves all attributes)
        filename_graphml = f'coauthorship_network_{period_name.replace("-", "_")}.graphml'
        nx.write_graphml(G_export, filename_graphml)
        print(f"Exported: {filename_graphml} ({G_export.number_of_nodes()} connected nodes)")
    except Exception as e:
        print(f"GraphML export failed: {str(e)}")

    try:
        # Export as GEXF (Gephi compatible)
        filename_gexf = f'coauthorship_network_{period_name.replace("-", "_")}.gexf'
        nx.write_gexf(G_export, filename_gexf)
        print(f"Exported: {filename_gexf} ({G_export.number_of_nodes()} connected nodes)")
    except Exception as e:
        print(f"GEXF export failed: {str(e)}")

# ================================================
# 9. ADVANCED FIGURE CUSTOMIZATION AND TIFF EXPORT
# ================================================

class NetworkVisualizationConfig:
    """
    Configuration class for customizing network visualizations.
    Allows fine-tuning of all visual parameters.
    """

    def __init__(self):
        # Figure dimensions and quality
        self.figure_width = 16
        self.figure_height = 12
        self.dpi = 300
        self.background_color = 'white'

        # Layout parameters
        self.layout_algorithm = 'spring'  # 'spring', 'circular', 'kamada_kawai'
        self.layout_k = 3  # Optimal distance between nodes
        self.layout_iterations = 50
        self.random_seed = 42

        # Node styling
        self.node_size_metric = 'pagerank'  # 'pagerank', 'degree', 'betweenness'
        self.node_size_multiplier = 3000
        self.node_size_base = 300
        self.node_alpha = 0.8
        self.node_border_width = 1
        self.node_border_color = 'black'

        # Edge styling
        self.edge_width_multiplier = 2
        self.edge_width_base = 0.5
        self.edge_alpha = 0.6
        self.edge_color = 'gray'
        self.show_edge_weights = False

        # Community colors
        self.community_colormap = 'Set3'  # 'Set3', 'tab10', 'Pastel1', 'Dark2'
        self.custom_colors = None  # List of custom colors

        # Labels
        self.show_labels = True
        self.label_font_size = 8
        self.label_font_weight = 'bold'
        self.label_font_color = 'black'
        self.max_label_length = 15
        self.top_n_labels = 50  # Show labels for top N influential authors only

        # Title and legend
        self.show_title = True
        self.title_font_size = 16
        self.title_font_weight = 'bold'
        self.show_legend = True
        self.legend_location = 'upper left'
        self.legend_bbox = (1, 1)

        # Export settings
        self.export_format = 'tiff'  # 'tiff', 'png', 'pdf', 'eps'
        self.export_dpi = 300
        self.export_bbox_inches = 'tight'
        self.export_facecolor = 'white'
        self.export_transparent = False

def create_custom_network_visualization(G, centralities, communities, period_name,
                                       top_authors_scores, config=None, output_dir="figures"):
    """
    Create highly customizable network visualization with TIFF export and isolated nodes removed.

    Args:
        G: NetworkX graph (may contain isolated nodes)
        centralities: centrality measures
        communities: community assignments
        period_name: time period name
        top_authors_scores: influence scores
        config: NetworkVisualizationConfig object
        output_dir: output directory for figures

    Returns:
        matplotlib figure object
    """

    # Use default config if none provided
    if config is None:
        config = NetworkVisualizationConfig()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f" Creating custom visualization for {period_name}...")

    # Remove isolated nodes for visualization
    G_viz = remove_isolated_nodes(G, "custom visualization")

    if G_viz.number_of_nodes() == 0:
        print(f" No connected nodes to visualize for {period_name}")
        return None

    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(config.figure_width, config.figure_height),
                          dpi=config.dpi, facecolor=config.background_color)

    # ========================================
    # CALCULATE LAYOUT
    # ========================================
    if config.layout_algorithm == 'spring':
        pos = nx.spring_layout(G_viz, k=config.layout_k, iterations=config.layout_iterations,
                              seed=config.random_seed)
    elif config.layout_algorithm == 'circular':
        pos = nx.circular_layout(G_viz)
    elif config.layout_algorithm == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G_viz)
    else:
        pos = nx.spring_layout(G_viz, seed=config.random_seed)

    # ========================================
    # PREPARE NODE ATTRIBUTES
    # ========================================
    node_sizes = []
    node_colors = []

    # Community colors
    unique_communities = sorted(set(communities.values()))
    if config.custom_colors and len(config.custom_colors) >= len(unique_communities):
        community_colors = config.custom_colors[:len(unique_communities)]
    else:
        cmap = plt.cm.get_cmap(config.community_colormap)
        community_colors = [cmap(i / max(1, len(unique_communities) - 1))
                           for i in range(len(unique_communities))]

    community_color_map = {comm: color for comm, color in zip(unique_communities, community_colors)}

    for node in G_viz.nodes():
        # Node size based on selected centrality measure
        if config.node_size_metric in centralities:
            centrality_score = centralities[config.node_size_metric].get(node, 0)
        else:
            centrality_score = centralities['pagerank'].get(node, 0)

        size = config.node_size_base + (centrality_score * config.node_size_multiplier)
        node_sizes.append(size)

        # Node color based on community
        community = communities.get(node, 0)
        node_colors.append(community_color_map[community])

    # ========================================
    # PREPARE EDGE ATTRIBUTES
    # ========================================
    edge_weights = [G_viz[u][v]['weight'] for u, v in G_viz.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [config.edge_width_multiplier * (weight / max_weight) + config.edge_width_base
                   for weight in edge_weights]

    # ========================================
    # DRAW NETWORK
    # ========================================

    # Draw edges
    nx.draw_networkx_edges(G_viz, pos,
                          width=edge_widths,
                          alpha=config.edge_alpha,
                          edge_color=config.edge_color,
                          ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G_viz, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=config.node_alpha,
                          linewidths=config.node_border_width,
                          edgecolors=config.node_border_color,
                          ax=ax)

    # ========================================
    # ADD LABELS
    # ========================================
    if config.show_labels:
        # Select top N authors for labeling (only connected ones)
        connected_top_authors = [(author, score) for author, score in top_authors_scores.items()
                               if author in G_viz.nodes()]
        top_n_authors = sorted(connected_top_authors, key=lambda x: x[1],
                              reverse=True)[:config.top_n_labels]

        labels = {}
        for author, _ in top_n_authors:
            if author in G_viz.nodes():
                name = G_viz.nodes[author]['name']
                if len(name) > config.max_label_length:
                    name = name[:config.max_label_length] + '...'
                labels[author] = name

        nx.draw_networkx_labels(G_viz, pos, labels,
                               font_size=config.label_font_size,
                               font_weight=config.label_font_weight,
                               font_color=config.label_font_color,
                               ax=ax)

    # ========================================
    # ADD TITLE
    # ========================================
    if config.show_title:
        isolated_count = G.number_of_nodes() - G_viz.number_of_nodes()
        title = (f'Co-authorship Network Analysis: {period_name}\n'
                f'Connected Nodes: {G_viz.number_of_nodes()}, Edges: {G_viz.number_of_edges()}, '
                f'Communities: {len(unique_communities)}')

        if isolated_count > 0:
            title += f'\n(Excluded {isolated_count} isolated nodes)'

        ax.set_title(title, fontsize=config.title_font_size,
                    fontweight=config.title_font_weight, pad=20)

    # ========================================
    # ADD LEGEND
    # ========================================
    if config.show_legend:
        legend_elements = []
        for comm, color in community_color_map.items():
            comm_size = sum(1 for c in communities.values() if c == comm)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=color, markersize=10,
                                             label=f'Community {comm+1} (n={comm_size})'))

        ax.legend(handles=legend_elements, loc=config.legend_location,
                 bbox_to_anchor=config.legend_bbox)

    # Remove axes for cleaner look
    ax.axis('off')

    # Tight layout
    plt.tight_layout()

    # ========================================
    # EXPORT FIGURE
    # ========================================
    filename = f'coauthorship_network_{period_name.replace("-", "_")}.{config.export_format}'
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath,
                format=config.export_format,
                dpi=config.export_dpi,
                bbox_inches=config.export_bbox_inches,
                facecolor=config.export_facecolor,
                transparent=config.export_transparent)

    print(f"Figure exported: {filepath}")

    return fig

def batch_export_all_networks(all_results, config=None, output_dir="figures"):
    """
    Export all network visualizations with consistent styling.

    Args:
        all_results: dictionary of analysis results for all periods
        config: NetworkVisualizationConfig object
        output_dir: output directory for figures
    """
    print(" Batch exporting all network visualizations...")

    exported_figures = {}

    for period_name, results in all_results.items():
        if results:
            try:
                fig = create_custom_network_visualization(
                    G=results['network'],
                    centralities=results['centralities'],
                    communities=results['communities'],
                    period_name=period_name,
                    top_authors_scores=dict(zip(results['top_authors'],
                                               [1] * len(results['top_authors']))),
                    config=config,
                    output_dir=output_dir
                )
                if fig:
                    exported_figures[period_name] = fig
                    plt.close(fig)  # Close to save memory

            except Exception as e:
                print(f"Error exporting {period_name}: {str(e)}")
                continue

    print(f"Exported {len(exported_figures)} network visualizations")
    return exported_figures

# ==========================================
# 10. COMPREHENSIVE EXPORT FUNCTIONS
# ==========================================

def export_all_network_data(all_results, csv_output_dir="network_exports",
                           figures_output_dir="figures"):
    """
    Export all network data and visualizations for all periods.

    Args:
        all_results: dictionary of analysis results
        csv_output_dir: directory for CSV exports
        figures_output_dir: directory for figure exports
    """
    print(" COMPREHENSIVE NETWORK DATA EXPORT")
    print("=" * 50)

    # Export CSV data for each period
    print("\n Exporting CSV data...")
    all_nodes_data = {}
    all_edges_data = {}

    for period_name, results in all_results.items():
        if results:
            nodes_df, edges_df = export_network_to_csv(
                G=results['full_network'],  # Use full network for complete CSV data
                centralities=results['centralities'],
                communities=results['communities'],
                period_name=period_name,
                output_dir=csv_output_dir
            )
            all_nodes_data[period_name] = nodes_df
            all_edges_data[period_name] = edges_df

    # Export summary statistics
    print("\n Exporting summary statistics...")
    summary_df = export_summary_statistics_csv(all_results, csv_output_dir)

    return all_nodes_data, all_edges_data, summary_df

# ================================================
# 11. EXPORTING WITH DIFFERENT CONFIGURATIONS
# ================================================

# Minimal style for presentation
def export_presentation_style(all_results):
    """Export figures with presentation-friendly styling."""
    config = NetworkVisualizationConfig()
    config.figure_width = 12
    config.figure_height = 10
    config.node_size_multiplier = 2000
    config.label_font_size = 10
    config.top_n_labels = 50 # set to large/small number to show more/less labels
    config.export_format = 'png'
    config.export_dpi = 150

    batch_export_all_networks(all_results, config,
     "/content/drive/My Drive/DATASETS/OBESITY.JOURNALS/figures_presentation")

# Publication style with custom colors
def export_publication_style(all_results):
    """Export figures with publication-ready styling."""
    config = NetworkVisualizationConfig()
    config.export_format = 'tiff'
    config.export_dpi = 600  # Ultra high quality
    config.figure_width = 16
    config.figure_height = 12
    config.community_colormap = 'Dark2'
    config.node_alpha = 0.9
    config.edge_alpha = 0.4
    config.label_font_size = 7 # Smaller font to fit more labels
    config.top_n_labels = 50 # set to large/small number to show more/less labels
    config.export_transparent = False
    config.export_facecolor = 'white'

    batch_export_all_networks(all_results, config,
      "/content/drive/My Drive/DATASETS/OBESITY.JOURNALS/figures_publication")

# Detailed analysis with edge weights shown
def export_detailed_analysis(all_results):
    """Export detailed figures for in-depth analysis."""
    config = NetworkVisualizationConfig()
    config.export_format = 'tiff'
    config.export_dpi = 300
    config.show_edge_weights = True
    config.edge_width_multiplier = 3
    config.node_size_multiplier = 4000
    config.top_n_labels = 15
    config.layout_algorithm = 'kamada_kawai'

    batch_export_all_networks(all_results, config,
          "/content/drive/My Drive/DATASETS/OBESITY.JOURNALS/figures_detailed")

# ===================================
# 12. COMPREHENSIVE ANALYSIS PIPELINE
# ===================================

def analyze_period(df, period_name, start_year, end_year):
    """
    Complete network analysis pipeline for a single time period.
    Updated to handle isolated nodes properly.

    Scientific Workflow:
    1. Data filtering and preprocessing
    2. Collaboration extraction
    3. Network construction
    4. Isolated node identification and handling
    5. Centrality analysis (connected nodes only)
    6. Top author selection (excludes isolated nodes)
    7. Community detection (connected subnetwork)
    8. Visualization (isolated nodes removed)
    9. Data export (complete data with isolation flags)

    This follows established bibliometric analysis procedures (Börner et al., 2004)
    """
    print(f"\n{'='*60}")
    print(f" ANALYZING PERIOD: {period_name} ({start_year}-{end_year})")
    print(f"{'='*60}")

    # Step 1: Filter data for time period
    print(f"\n Filtering data for {period_name}...")
    df_period = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
    print(f"Period dataset: {len(df_period):,} publications")

    if len(df_period) == 0:
        print(f" No data for period {period_name}")
        return None

    # Step 2: Extract collaborations
    print(f"\n Extracting author collaborations...")
    collaborations, author_info = extract_author_collaborations(df_period)

    if len(collaborations) == 0:
        print(f" No collaborations found for period {period_name}")
        return None

    # Step 3: Build network
    print(f"\n Constructing co-authorship network...")
    G_full = build_coauthorship_network(collaborations, author_info)

    # Step 4: Analyze isolated nodes
    isolated_nodes = list(nx.isolates(G_full))
    print(f"\n Isolated node analysis...")
    print(f" Found {len(isolated_nodes):,} isolated nodes")
    print(f"Connected nodes: {G_full.number_of_nodes() - len(isolated_nodes):,}")

    # Step 5: Calculate centralities (connected nodes only)
    print(f"\n Calculating centrality measures...")
    centralities = calculate_centrality_measures(G_full)

    # Step 6: Select top authors (automatically excludes isolated nodes)
    print(f"\n Selecting top 50 influential authors...")
    top_authors, top_scores = select_top_authors(G_full, centralities, top_n=50)

    if not top_authors:
        print(f" No top authors found for period {period_name}")
        return None

    # Step 7: Create focused subnetwork (connected only)
    print(f"\n Creating focused subnetwork...")
    subG = create_subnetwork(G_full, top_authors)

    if subG.number_of_nodes() == 0:
        print(f" Empty subnetwork for period {period_name}")
        return None

    # Step 8: Community detection (connected subnetwork)
    print(f"\n Detecting communities...")
    communities, modularity = detect_communities_walktrap(subG)

    # Step 9: Visualization (isolated nodes automatically excluded)
    print(f"\n Creating visualization...")
    fig = visualize_network(subG, centralities, communities, period_name, top_scores)

    # Step 10: Export data (full network with isolation flags)
    print(f"\n Exporting network data...")
    export_network_data(subG, communities, centralities, period_name)

    # Return analysis results
    results = {
        'period': period_name,
        'network': subG,  # Connected subnetwork for visualization/analysis
        'full_network': G_full,  # Complete network including isolated nodes for CSV export
        'centralities': centralities,
        'communities': communities,
        'modularity': modularity,
        'top_authors': top_authors,
        'isolated_nodes': isolated_nodes,
        'isolated_count': len(isolated_nodes),
        'visualization': fig
    }

    print(f"\n Analysis complete for {period_name}")
    print(f"Analyzed network: {subG.number_of_nodes():,} connected nodes")
    print(f" Excluded from visualization: {len(isolated_nodes):,} isolated nodes")

    return results

# =============================================
# 13. MAIN EXECUTION: TEMPORAL NETWORK ANALYSIS
# =============================================

def main_analysis():
    """
    Execute complete temporal co-authorship network analysis.
    Updated to properly handle isolated nodes.

    Scientific Approach:
    - Systematic temporal slicing for evolutionary analysis
    - Consistent methodology across all periods
    - Proper handling of isolated nodes (excluded from visualization, included in data)
    - Comprehensive documentation for reproducibility
    - Publication-ready outputs
    """
    print(" STARTING COMPREHENSIVE CO-AUTHORSHIP NETWORK ANALYSIS")
    print("=" * 80)
    print("Scientific Method: Temporal Evolution of Scientific Collaboration")
    print("Algorithm: Walktrap Community Detection on Weighted Networks")
    print("Layout: Fruchterman-Reingold Force-Directed Positioning")
    print("Isolated Nodes: Excluded from visualization, included in data exports")
    print("=" * 80)

    # Store all results
    all_results = {}

    # Analyze each time period
    for period_name, (start_year, end_year) in periods.items():
        try:
            results = analyze_period(df, period_name, start_year, end_year)
            if results:
                all_results[period_name] = results
        except Exception as e:
            logger.error(f"Error analyzing period {period_name}: {str(e)}")
            continue

    # Summary statistics across periods
    print(f"\n{'='*60}")
    print(" TEMPORAL ANALYSIS SUMMARY")
    print(f"{'='*60}")

    for period, results in all_results.items():
        if results:
            print(f"\n{period}:")
            print(f"   • Total authors: {results['full_network'].number_of_nodes():,}")
            print(f"   • Isolated authors: {results['isolated_count']:,}")
            print(f"   • Connected authors: {results['network'].number_of_nodes():,}")
            print(f"   • Collaborations: {results['network'].number_of_edges():,}")
            print(f"   • Communities: {len(set(results['communities'].values()))}")
            print(f"   • Modularity: {results['modularity']:.3f}")

    print(f"\n ANALYSIS COMPLETE!")
    print(f" Generated files per period:")
    print(f"   • Network visualization (PNG) - connected nodes only")
    print(f"   • GraphML export for Gephi/Cytoscape - connected nodes only")
    print(f"   • GEXF export for external analysis - connected nodes only")
    print(f"   • CSV exports include isolated node flags for complete data")

    return all_results

# ==========================================
# 14. EXECUTE COMPREHENSIVE EXPORT
# ==========================================

def run_complete_export(all_results):
    """Run complete export of all data and visualizations."""
    print(" STARTING COMPREHENSIVE EXPORT PROCESS")
    print("=" * 60)

    # Export all CSV data
    print("\n EXPORTING CSV DATA...")
    nodes_data, edges_data, summary_data = export_all_network_data(all_results)

    # Export multiple figure styles
    print("\n EXPORTING FIGURES...")

    # High-quality TIFF for publication
    print("\n    Publication quality TIFF...")
    export_publication_style(all_results)

    # Presentation style
    print("\n    Presentation style...")
    export_presentation_style(all_results)

    # High-resolution detailed analysis
    print("\n    Detailed analysis style...")
    export_detailed_analysis(all_results)

    print("\n COMPLETE EXPORT FINISHED!")
    print(" Generated file structure:")
    print("   network_exports/")
    print("   ├── nodes_[period].csv (includes isolated node flags)")
    print("   ├── edges_[period].csv (connected nodes only)")
    print("   └── network_summary_statistics.csv")
    print("   figures_publication/")
    print("   ├── coauthorship_network_[period].tiff")
    print("   figures_presentation/")
    print("   ├── coauthorship_network_[period].png")
    print("   figures_detailed/")
    print("   └── coauthorship_network_[period].tiff")

# ==========================================
# 15. CUSTOM CONFIGURATION EXAMPLE
# ==========================================

def create_custom_export(all_results):
    """Example of creating completely custom visualization settings."""

    # Create custom configuration
    custom_config = NetworkVisualizationConfig()

    # Customize all parameters
    custom_config.figure_width = 24
    custom_config.figure_height = 18
    custom_config.dpi = 300
    custom_config.export_format = 'tiff'
    custom_config.export_dpi = 600

    # Node styling
    custom_config.node_size_metric = 'betweenness'  # Use betweenness centrality
    custom_config.node_size_multiplier = 5000
    custom_config.node_alpha = 0.85
    custom_config.node_border_width = 2

    # Edge styling
    custom_config.edge_width_multiplier = 4
    custom_config.edge_alpha = 0.3
    custom_config.edge_color = '#2E2E2E'

    # Colors - use custom color palette
    custom_config.custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                                  '#FFEAA7', '#DDA0DD', '#FFB347', '#87CEEB']

    # Labels
    custom_config.label_font_size = 9
    custom_config.top_n_labels = 50
    custom_config.max_label_length = 20 # Shorter names to reduce clutter

    # Layout
    custom_config.layout_algorithm = 'spring'
    custom_config.layout_k = 4
    custom_config.layout_iterations = 100

    # Export with custom configuration
    batch_export_all_networks(all_results, custom_config, "figures_custom")
    print(" Custom styled figures exported!")

# ==========================================
# 16. EXECUTE ANALYSIS AND EXPORT
# ==========================================

# Run the complete analysis
print(" EXECUTING COMPLETE NETWORK ANALYSIS...")
results = main_analysis()

# Export all data and visualizations
print("\n EXECUTING COMPREHENSIVE EXPORT...")
run_complete_export(results)

# Create custom export example
print("\n CREATING CUSTOM EXPORT EXAMPLE...")
create_custom_export(results)

# ==========================================
# 17. SCIENTIFIC REFERENCES AND JUSTIFICATION
# ==========================================
"""
 THEORETICAL FOUNDATIONS:

1. CO-AUTHORSHIP NETWORKS IN SCIENCE:
   • Newman, M.E.J. (2001). Scientific collaboration networks. PNAS, 98(2), 404-409.
     → Establishes co-authorship as proxy for scientific collaboration

   • Barabási, A.-L., et al. (2002). Evolution of scientific collaborations.
     Physica A, 311(3-4), 590-614.
     → Demonstrates temporal evolution patterns in collaboration networks

2. CENTRALITY MEASURES FOR SCIENTIFIC INFLUENCE:
   • Freeman, L.C. (1977). Betweenness centrality as measure of influence.
     → Identifies brokers and boundary spanners in networks

   • Page, L., et al. (1999). PageRank: Bringing order to the web.
     → Prestige-based ranking suitable for scientific influence

3. COMMUNITY DETECTION:
   • Pons, P., & Latapy, M. (2005). Computing communities using random walks.
     → Walktrap algorithm particularly effective for social networks

   • Newman, M.E.J. (2006). Modularity and community structure in networks.
     → Modularity as quality measure for community detection

4. NETWORK VISUALIZATION:
   • Fruchterman, T.M.J., & Reingold, E.M. (1991). Graph drawing by
     force-directed placement. Software: Practice and Experience, 21(11).
     → Aesthetic and interpretable network layouts

5. TEMPORAL BIBLIOMETRIC ANALYSIS:
   • Glänzel, W. (2003). Bibliometrics as research field. Course handouts.
     → Methodological foundations for temporal analysis

   • Börner, K., Chen, C., & Boyack, K.W. (2004). Visualizing knowledge domains.
     Annual Review of Information Science and Technology, 37(1), 179-255.
     → Best practices for longitudinal bibliometric studies

6. ISOLATED NODES IN NETWORK ANALYSIS:
   • Scott, J. (2017). Social Network Analysis. 4th ed. SAGE Publications.
     → Theoretical justification for excluding isolated nodes from visualization

   • Wasserman, S., & Faust, K. (1994). Social Network Analysis: Methods and Applications.
     → Methodological guidance on handling disconnected components

METHODOLOGICAL CHOICES:

• Temporal Slicing: 8-year periods balance statistical power with temporal resolution
• Top 50 Authors: Focuses on most influential while maintaining network connectivity
• Walktrap Algorithm: Random walk-based method ideal for collaboration networks
• Multiple Centralities: Captures different dimensions of scientific influence
• Force-directed Layout: Optimizes readability and interpretation
• Isolated Node Exclusion: Removes non-collaborative authors from visualization while preserving complete data

RESEARCH APPLICATIONS:

This methodology enables investigation of:
• Evolution of research collaboration patterns
• Identification of key scientific actors and brokers
• Emergence and dissolution of research communities
• Temporal shifts in collaboration strategies
• Network-based prediction of future collaborations
• Impact of isolated vs. collaborative research approaches

DATA INTEGRITY:

• Complete Data Preservation: All authors included in CSV exports with isolation flags
• Visualization Focus: Only connected authors shown for meaningful collaboration analysis
• Transparency: Clear documentation of excluded nodes in all outputs
• Reproducibility: Consistent methodology across all temporal periods
"""
# -*- coding: utf-8 -*-
"""
Obesity research in specialty journals from 2000 to 2023: A bibliometric analysis
"""

!pip install python-igraph

import os
import warnings
from google.colab import drive
# Core data manipulation and analysis
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import itertools
import ast
from tqdm import tqdm
# Network analysis libraries
import networkx as nx
import igraph as ig
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
print(f"NetworkX version: {nx.__version__}")
print(f"igraph version: {ig.__version__}")

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for high-resolution output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Mount Google Drive
drive.mount('/content/drive')

# Define input/output directories
input_dir = '/content/drive/My Drive/DATASETS/OBESITY.JOURNALS/'
output_dir = '/content/drive/My Drive/DATASETS/OBESITY.JOURNALS/concepts_network'
os.makedirs(output_dir, exist_ok=True)

# Define temporal periods for analysis
periods = {
    '2000-2007': (2000, 2007),
    '2008-2015': (2008, 2015),
    '2016-2023': (2016, 2023)
}

# Analysis parameters
config = {
    'min_concept_frequency': 5,        # Minimum frequency for concept inclusion
    'min_cooccurrence_weight': 2,      # Minimum edge weight for inclusion
    'top_concepts_viz': 50,            # Number of top concepts to visualize
    'use_relevance_weighting': True,   # Whether to weight edges by relevance scores
    'relevance_threshold': 0.3,        # Minimum relevance score for concept inclusion
    'centrality_measure': 'degree',    # 'degree', 'betweenness', or 'pagerank'
    'community_resolution': 1.0,       # Resolution parameter for community detection
    'layout_iterations': 1000,         # Iterations for Fruchterman-Reingold layout
    'random_seed': 42                  # For reproducibility
}

# Set random seeds for reproducibility
np.random.seed(config['random_seed'])

print("Configuration Parameters:")
for key, value in config.items():
    print(f"  {key}: {value}")

print(f"\n Analysis Periods:")
for period_name, (start, end) in periods.items():
    print(f"  {period_name}: {start}-{end}")

def safe_eval(x):
    """Safely evaluate string representations of lists."""
    if pd.isna(x) or x == '' or x == '[]':
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return x if isinstance(x, list) else []

def normalize_concept(concept):
    """Normalize concept names for consistency."""
    if pd.isna(concept) or concept == '':
        return None
    return str(concept).lower().strip()

def extract_concepts_with_scores(concepts_str, scores_str, relevance_threshold=0.3):
    """
    Extract concepts and their relevance scores, filtering by threshold.

    Parameters:
    -----------
    concepts_str : str or list
        String representation or list of concepts
    scores_str : str or list
        String representation or list of concept score dictionaries
    relevance_threshold : float
        Minimum relevance score for inclusion

    Returns:
    --------
    tuple: (filtered_concepts, concepts_scores_dict)
    """
    concepts = safe_eval(concepts_str)
    scores = safe_eval(scores_str)

    if not concepts or not scores:
        return [], {}

    # Create concept-score mapping
    concepts_scores = {}
    for score_dict in scores:
        if isinstance(score_dict, dict) and 'concept' in score_dict and 'relevance' in score_dict:
            concept = normalize_concept(score_dict['concept'])
            relevance = score_dict.get('relevance', 0)
            if concept and relevance >= relevance_threshold:
                concepts_scores[concept] = relevance

    # Filter concepts by relevance threshold
    filtered_concepts = []
    for concept in concepts:
        normalized = normalize_concept(concept)
        if normalized and normalized in concepts_scores:
            filtered_concepts.append(normalized)

    return filtered_concepts, concepts_scores

def filter_dataframe_by_period(df, period_name, periods):
    """Filter dataframe by temporal period."""
    start_year, end_year = periods[period_name]
    filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
    print(f" {period_name}: {len(filtered_df):,} records ({start_year}-{end_year})")
    return filtered_df

print("Data preprocessing functions defined")

def extract_cooccurrences(df, use_relevance_weighting=True, min_concepts=2):
    """
    Extract concept co-occurrences from the dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with 'concepts' and 'concepts_scores' columns
    use_relevance_weighting : bool
        Whether to weight co-occurrences by relevance scores
    min_concepts : int
        Minimum number of concepts per document to consider

    Returns:
    --------
    dict: Edge weights for concept pairs
    dict: Concept frequencies
    dict: Average relevance scores per concept
    """
    cooccurrence_weights = defaultdict(float)
    concept_frequencies = defaultdict(int)
    concept_relevance_scores = defaultdict(list)

    valid_documents = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
        # Extract concepts and scores
        concepts, scores_dict = extract_concepts_with_scores(
            row['concepts'],
            row['concepts_scores'],
            config['relevance_threshold']
        )

        if len(concepts) < min_concepts:
            continue

        valid_documents += 1

        # Update concept frequencies and relevance scores
        for concept in concepts:
            concept_frequencies[concept] += 1
            if concept in scores_dict:
                concept_relevance_scores[concept].append(scores_dict[concept])

        # Generate concept pairs and update co-occurrence weights
        for concept1, concept2 in itertools.combinations(concepts, 2):
            # Ensure consistent ordering (alphabetical)
            if concept1 > concept2:
                concept1, concept2 = concept2, concept1

            # Calculate edge weight
            if use_relevance_weighting and concept1 in scores_dict and concept2 in scores_dict:
                # Weight by average relevance of the two concepts
                weight = (scores_dict[concept1] + scores_dict[concept2]) / 2
            else:
                weight = 1.0

            cooccurrence_weights[(concept1, concept2)] += weight

    # Calculate average relevance scores
    avg_relevance_scores = {}
    for concept, scores in concept_relevance_scores.items():
        avg_relevance_scores[concept] = np.mean(scores) if scores else 0.0

    print(f"Processed {valid_documents:,} valid documents")
    print(f"Found {len(concept_frequencies):,} unique concepts")
    print(f"Generated {len(cooccurrence_weights):,} concept pairs")

    return dict(cooccurrence_weights), dict(concept_frequencies), avg_relevance_scores

def build_network(cooccurrence_weights, concept_frequencies, min_frequency=5, min_weight=2):
    """
    Build NetworkX graph from co-occurrence data.

    Parameters:
    -----------
    cooccurrence_weights : dict
        Edge weights for concept pairs
    concept_frequencies : dict
        Frequency of each concept
    min_frequency : int
        Minimum concept frequency for inclusion
    min_weight : float
        Minimum edge weight for inclusion

    Returns:
    --------
    networkx.Graph: The constructed network
    """
    # Filter concepts by frequency
    valid_concepts = {concept for concept, freq in concept_frequencies.items()
                     if freq >= min_frequency}

    print(f"Concepts after frequency filtering (>= {min_frequency}): {len(valid_concepts):,}")

    # Create graph
    G = nx.Graph()

    # Add nodes with attributes
    for concept in valid_concepts:
        G.add_node(concept,
                  frequency=concept_frequencies[concept],
                  label=concept)

    # Add edges with weights
    edges_added = 0
    for (concept1, concept2), weight in cooccurrence_weights.items():
        if (concept1 in valid_concepts and
            concept2 in valid_concepts and
            weight >= min_weight):
            G.add_edge(concept1, concept2, weight=weight)
            edges_added += 1

    print(f"Final network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Network density: {nx.density(G):.4f}")

    return G

print("Network construction functions defined")

def calculate_centrality_measures(G):
    """Calculate various centrality measures for the network."""
    print("Calculating centrality measures...")

    centrality_measures = {}

    # Degree centrality
    centrality_measures['degree'] = nx.degree_centrality(G)

    # Betweenness centrality (for smaller networks due to computational complexity)
    if G.number_of_nodes() <= 1000:
        centrality_measures['betweenness'] = nx.betweenness_centrality(G, weight='weight')
    else:
        print("Skipping betweenness centrality (network too large)")
        centrality_measures['betweenness'] = {}

    # PageRank centrality
    centrality_measures['pagerank'] = nx.pagerank(G, weight='weight')

    # Weighted degree (strength)
    centrality_measures['weighted_degree'] = dict(G.degree(weight='weight'))

    return centrality_measures

def detect_communities_walktrap(G):
    """
    Apply Walktrap community detection using igraph.

    Parameters:
    -----------
    G : networkx.Graph
        The network to analyze

    Returns:
    --------
    dict: Node to community mapping
    igraph.Graph: igraph version of the network
    """
    print("Applying Walktrap community detection...")

    # Convert NetworkX to igraph
    edge_list = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    node_list = list(G.nodes())

    # Create igraph graph
    ig_graph = ig.Graph()
    ig_graph.add_vertices(node_list)
    ig_graph.add_edges([(node_list.index(u), node_list.index(v)) for u, v, w in edge_list])
    ig_graph.es['weight'] = [w for u, v, w in edge_list]
    ig_graph.vs['name'] = node_list

    # Apply Walktrap algorithm
    walktrap = ig_graph.community_walktrap(weights='weight', steps=4)
    communities = walktrap.as_clustering()

    # Create node to community mapping
    community_mapping = {}
    for i, community_id in enumerate(communities.membership):
        node_name = node_list[i]
        community_mapping[node_name] = community_id

    print(f"Found {len(communities):,} communities")
    print(f"Modularity: {communities.modularity:.4f}")

    # Print community sizes
    community_sizes = Counter(communities.membership)
    print("Community sizes:")
    for comm_id, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f" Community {comm_id}: {size} nodes")

    return community_mapping, ig_graph

def get_top_concepts(G, centrality_measures, community_mapping, n_top=50, centrality_type='degree'):
    """
    Get top N concepts based on centrality measure.

    Parameters:
    -----------
    G : networkx.Graph
        The network
    centrality_measures : dict
        Dictionary of centrality measures
    community_mapping : dict
        Node to community mapping
    n_top : int
        Number of top concepts to return
    centrality_type : str
        Type of centrality to use for ranking

    Returns:
    --------
    list: Top concepts with their attributes
    """
    centrality_scores = centrality_measures.get(centrality_type, {})

    if not centrality_scores:
        print(f"{centrality_type} centrality not available, using degree centrality")
        centrality_scores = centrality_measures['degree']

    # Create list of concept data
    concept_data = []
    for node in G.nodes():
        concept_data.append({
            'concept': node,
            'centrality': centrality_scores.get(node, 0),
            'frequency': G.nodes[node].get('frequency', 0),
            'community': community_mapping.get(node, -1),
            'degree': G.degree(node),
            'weighted_degree': centrality_measures['weighted_degree'].get(node, 0)
        })

    # Sort by centrality and take top N
    concept_data.sort(key=lambda x: x['centrality'], reverse=True)
    top_concepts = concept_data[:n_top]

    print(f"Top {n_top} concepts by {centrality_type} centrality:")
    for i, concept in enumerate(top_concepts[:10]):
        print(f"   {i+1:2d}. {concept['concept']:<30} "
              f"(centrality: {concept['centrality']:.4f}, "
              f"frequency: {concept['frequency']:4d}, "
              f"community: {concept['community']:2d})")

    return top_concepts

print("Network analysis functions defined")

def create_network_visualization(G, top_concepts, community_mapping, period_name,
                                output_dir, layout_iterations=1000):
    """
    Create publication-quality network visualization.

    Parameters:
    -----------
    G : networkx.Graph
        The network to visualize
    top_concepts : list
        List of top concepts to include in visualization
    community_mapping : dict
        Node to community mapping
    period_name : str
        Name of the time period
    output_dir : str
        Directory to save the visualization
    layout_iterations : int
        Number of iterations for layout algorithm
    """
    print(f"Creating network visualization for {period_name}...")

    # Create subgraph with top concepts
    top_concept_names = [c['concept'] for c in top_concepts]
    subgraph = G.subgraph(top_concept_names).copy()

    if subgraph.number_of_nodes() == 0:
        print("No nodes to visualize")
        return

    # Calculate layout using Fruchterman-Reingold
    print("Calculating Fruchterman-Reingold layout...")
    pos = nx.spring_layout(subgraph, iterations=layout_iterations,
                          weight='weight', k=3, seed=config['random_seed'])

    # Prepare node attributes for visualization
    communities = [community_mapping.get(node, -1) for node in subgraph.nodes()]
    unique_communities = list(set(communities))
    n_communities = len(unique_communities)

    # Generate PASTEL/PASTRY colors for communities
    if n_communities <= 10:
        colors = sns.color_palette("pastel", n_communities)
    else:
        colors = sns.color_palette("Set3", n_communities)

    community_colors = {comm: colors[i] for i, comm in enumerate(unique_communities)}
    node_colors = [community_colors[community_mapping.get(node, -1)] for node in subgraph.nodes()]

    # Calculate node sizes based on degree
    degrees = dict(subgraph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [300 + (degrees.get(node, 0) / max_degree) * 700 for node in subgraph.nodes()]

    # Calculate edge widths based on weights
    edge_weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + (weight / max_weight) * 3 for weight in edge_weights]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))  # Larger figure for better label visibility

    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.3,
                          edge_color='gray', ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                          node_size=node_sizes, alpha=0.8, ax=ax,
                          edgecolors='black', linewidths=0.5)  # Add black edges to nodes

    # Draw labels for ALL nodes (all 50 nodes)
    nx.draw_networkx_labels(subgraph, pos, font_size=7, font_weight='bold',
                           ax=ax, bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor='white', alpha=0.7, edgecolor='none'))

    # Create legend for communities
    legend_elements = [Patch(facecolor=community_colors[comm],
                           label=f'Community {comm}')
                      for comm in sorted(unique_communities)[:15]]  # Show top 15 communities

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Set title and labels
    ax.set_title(f'Concept Co-occurrence Network: {period_name}\n'
                f'({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)',
                fontsize=18, fontweight='bold', pad=20)

    ax.axis('off')
    plt.tight_layout()

    # Save the plot
    filename = f'concept_network_{period_name.replace("-", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved: {filepath}")

    plt.show()

    return fig

def create_centrality_comparison_plot(all_period_results, output_dir):
    """Create comparison plot of centrality measures across periods."""
    print("Creating centrality comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    centrality_types = ['degree', 'pagerank', 'weighted_degree']

    for i, centrality_type in enumerate(centrality_types):
        ax = axes[i]

        # Collect top concepts across all periods for this centrality measure
        all_concepts = set()
        for period_data in all_period_results.values():
            if centrality_type in period_data.get('centrality_measures', {}):
                centrality_scores = period_data['centrality_measures'][centrality_type]
                top_concepts = sorted(centrality_scores.items(),
                                    key=lambda x: x[1], reverse=True)[:15]
                all_concepts.update([concept for concept, _ in top_concepts])

        # Create comparison data
        comparison_data = []
        for period_name, period_data in all_period_results.items():
            if centrality_type in period_data.get('centrality_measures', {}):
                centrality_scores = period_data['centrality_measures'][centrality_type]
                for concept in all_concepts:
                    comparison_data.append({
                        'period': period_name,
                        'concept': concept,
                        'centrality': centrality_scores.get(concept, 0)
                    })

        if not comparison_data:
            continue

        df_comp = pd.DataFrame(comparison_data)

        # Create heatmap
        pivot_data = df_comp.pivot(index='concept', columns='period', values='centrality')
        pivot_data = pivot_data.fillna(0)

        # Sort by average centrality
        pivot_data['avg'] = pivot_data.mean(axis=1)
        pivot_data = pivot_data.sort_values('avg', ascending=False).head(15)
        pivot_data = pivot_data.drop('avg', axis=1)

        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title(f'{centrality_type.title()} Centrality Comparison', fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Concept')

    # Remove empty subplot
    if len(centrality_types) < 4:
        fig.delaxes(axes[3])

    plt.tight_layout()

    # Save the plot
    filename = 'centrality_comparison_across_periods.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Centrality comparison saved: {filepath}")

    plt.show()

    return fig

print("Visualization functions defined")

def analyze_period(df, period_name, periods, config, output_dir):
    """
    Complete analysis pipeline for a single time period.

    Parameters:
    -----------
    df : pandas.DataFrame
        Complete dataset
    period_name : str
        Name of the period to analyze
    periods : dict
        Dictionary of period definitions
    config : dict
        Configuration parameters
    output_dir : str
        Output directory for results

    Returns:
    --------
    dict: Complete analysis results for the period
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING PERIOD: {period_name}")
    print(f"{'='*60}")

    # Filter data by period
    period_df = filter_dataframe_by_period(df, period_name, periods)

    if len(period_df) == 0:
        print(f"No data available for period {period_name}")
        return {}

    # Extract co-occurrences
    cooccurrence_weights, concept_frequencies, avg_relevance_scores = extract_cooccurrences(
        period_df,
        use_relevance_weighting=config['use_relevance_weighting'],
        min_concepts=2
    )

    if not cooccurrence_weights:
        print(f"No co-occurrences found for period {period_name}")
        return {}

    # Build network
    G = build_network(
        cooccurrence_weights,
        concept_frequencies,
        min_frequency=config['min_concept_frequency'],
        min_weight=config['min_cooccurrence_weight']
    )

    if G.number_of_nodes() == 0:
        print(f"Empty network for period {period_name}")
        return {}

    # Calculate centrality measures
    centrality_measures = calculate_centrality_measures(G)

    # Detect communities
    community_mapping, ig_graph = detect_communities_walktrap(G)

    # Get top concepts
    top_concepts = get_top_concepts(
        G, centrality_measures, community_mapping,
        n_top=config['top_concepts_viz'],
        centrality_type=config['centrality_measure']
    )

    # Create visualization
    fig = create_network_visualization(
        G, top_concepts, community_mapping, period_name,
        output_dir, config['layout_iterations']
    )

    # Prepare results
    results = {
        'period_name': period_name,
        'network': G,
        'centrality_measures': centrality_measures,
        'community_mapping': community_mapping,
        'top_concepts': top_concepts,
        'concept_frequencies': concept_frequencies,
        'avg_relevance_scores': avg_relevance_scores,
        'cooccurrence_weights': cooccurrence_weights,
        'ig_graph': ig_graph,
        'visualization': fig,
        'stats': {
            'n_documents': len(period_df),
            'n_concepts': len(concept_frequencies),
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'n_communities': len(set(community_mapping.values()))
        }
    }

    print(f"Analysis complete for {period_name}")
    return results

def export_results(results, output_dir):
    """Export detailed results for each period."""
    print("\n Exporting detailed results...")

    for period_name, period_results in results.items():
        print(f"Exporting {period_name}...")

        # Create period-specific directory
        period_dir = os.path.join(output_dir, period_name.replace("-", "_"))
        os.makedirs(period_dir, exist_ok=True)
        print(f"   Created directory: {period_dir}")

        try:
            # Export top concepts
            if 'top_concepts' in period_results:
                top_concepts_df = pd.DataFrame(period_results['top_concepts'])
                top_concepts_path = os.path.join(period_dir, 'top_concepts.csv')
                top_concepts_df.to_csv(top_concepts_path, index=False)
                print(f" Exported top_concepts.csv ({len(top_concepts_df)} concepts)")

            # Export network as GraphML
            if 'network' in period_results:
                network_path = os.path.join(period_dir, 'network.graphml')
                nx.write_graphml(period_results['network'], network_path)
                print(f" Exported network.graphml ({period_results['network'].number_of_nodes()} nodes)")

            # Export network as edge list (more universal format)
            if 'network' in period_results:
                G = period_results['network']
                edge_list = []
                for u, v, data in G.edges(data=True):
                    edge_list.append({
                        'source': u,
                        'target': v,
                        'weight': data.get('weight', 1)
                    })
                edge_df = pd.DataFrame(edge_list)
                edge_path = os.path.join(period_dir, 'network_edges.csv')
                edge_df.to_csv(edge_path, index=False)
                print(f" Exported network_edges.csv ({len(edge_df)} edges)")

                # Export node attributes
                node_list = []
                for node in G.nodes():
                    node_list.append({
                        'node': node,
                        'frequency': G.nodes[node].get('frequency', 0),
                        'community': period_results['community_mapping'].get(node, -1),
                        'degree': G.degree(node)
                    })
                node_df = pd.DataFrame(node_list)
                node_path = os.path.join(period_dir, 'network_nodes.csv')
                node_df.to_csv(node_path, index=False)
                print(f" Exported network_nodes.csv ({len(node_df)} nodes)")

            # Export community assignments
            if 'community_mapping' in period_results:
                community_df = pd.DataFrame([
                    {'concept': concept, 'community': community}
                    for concept, community in period_results['community_mapping'].items()
                ])
                community_path = os.path.join(period_dir, 'communities.csv')
                community_df.to_csv(community_path, index=False)
                print(f" Exported communities.csv ({len(community_df)} concepts)")

            # Export centrality measures
            if 'centrality_measures' in period_results:
                centrality_data = []
                for node in period_results['network'].nodes():
                    row = {'concept': node}
                    for measure_name, measure_dict in period_results['centrality_measures'].items():
                        row[f'{measure_name}_centrality'] = measure_dict.get(node, 0)
                    centrality_data.append(row)

                centrality_df = pd.DataFrame(centrality_data)
                centrality_path = os.path.join(period_dir, 'centrality_measures.csv')
                centrality_df.to_csv(centrality_path, index=False)
                print(f" Exported centrality_measures.csv ({len(centrality_df)} concepts)")

            # Export concept frequencies and relevance scores
            if 'concept_frequencies' in period_results:
                freq_data = []
                for concept, frequency in period_results['concept_frequencies'].items():
                    freq_data.append({
                        'concept': concept,
                        'frequency': frequency,
                        'avg_relevance': period_results.get('avg_relevance_scores', {}).get(concept, 0)
                    })
                freq_df = pd.DataFrame(freq_data)
                freq_df = freq_df.sort_values('frequency', ascending=False)
                freq_path = os.path.join(period_dir, 'concept_frequencies.csv')
                freq_df.to_csv(freq_path, index=False)
                print(f" Exported concept_frequencies.csv ({len(freq_df)} concepts)")

            # Export co-occurrence weights (top 1000 pairs)
            if 'cooccurrence_weights' in period_results:
                cooc_data = []
                for (concept1, concept2), weight in period_results['cooccurrence_weights'].items():
                    cooc_data.append({
                        'concept1': concept1,
                        'concept2': concept2,
                        'weight': weight
                    })
                cooc_df = pd.DataFrame(cooc_data)
                cooc_df = cooc_df.sort_values('weight', ascending=False).head(1000)  # Top 1000 pairs
                cooc_path = os.path.join(period_dir, 'top_cooccurrences.csv')
                cooc_df.to_csv(cooc_path, index=False)
                print(f" Exported top_cooccurrences.csv ({len(cooc_df)} pairs)")

        except Exception as e:
            print(f" Error exporting files for {period_name}: {str(e)}")
            continue

        print(f"Successfully exported all files for {period_name}")

# Load and check the data is loaded
file_name = os.path.join(input_dir, 'merged_results_filtered.csv')
df = pd.read_csv(file_name)
print(f" Dataset info:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Years available: {df['year'].min()} - {df['year'].max()}")

# Check for required columns
required_columns = ['concepts', 'concepts_scores', 'year']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing required columns: {missing_columns}")
else:
    print("All required columns present")

def run_complete_analysis(df, periods, config, output_dir):
    """
    Run the complete analysis pipeline for all periods.
    (Updated version with proper export calling)
    """
    print(f"\n STARTING COMPLETE NETWORK ANALYSIS")
    print(f"Dataset shape: {df.shape}")
    print(f"Analyzing {len(periods)} time periods")

    all_results = {}

    # Analyze each period
    for period_name in periods.keys():
        try:
            results = analyze_period(df, period_name, periods, config, output_dir)
            if results:
                all_results[period_name] = results
        except Exception as e:
            print(f"Error analyzing period {period_name}: {str(e)}")
            continue

    # Create comparison visualizations
    if len(all_results) > 1:
        try:
            create_centrality_comparison_plot(all_results, output_dir)
        except Exception as e:
            print(f" Could not create comparison plot: {str(e)}")

    # Save summary results
    summary_stats = {}
    for period_name, results in all_results.items():
        summary_stats[period_name] = results['stats']

    summary_df = pd.DataFrame(summary_stats).T
    summary_filepath = os.path.join(output_dir, 'analysis_summary.csv')
    summary_df.to_csv(summary_filepath)
    print(f"Summary statistics saved: {summary_filepath}")

    # Export detailed results for each period
    export_results(all_results, output_dir)

    # Display summary
    print(f"\n ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(summary_df)

    return all_results

print("Main analysis pipeline defined")

# Run the complete analysis
results = run_complete_analysis(df, periods, config, output_dir)

# Export all results
export_results(results, output_dir)

# List all exported files
print(f"\n EXPORTED FILES:")
for period_name in results.keys():
    period_dir = os.path.join(output_dir, period_name.replace("-", "_"))
    if os.path.exists(period_dir):
        files = os.listdir(period_dir)
        print(f"\n  {period_name}:")
        for file in sorted(files):
            print(f"    {file}")

print(f"\n ANALYSIS COMPLETED")
print(f"Results saved to: {output_dir}")
print(f"Analyzed {len(results)} periods successfully")

""" INFORMATION ABOUT THE METHODS EMPLOYED:
* Walktrap Algorithm: Excellent for detecting communities in weighted networks, especially effective for concept co-occurrence networks (Pons & Latapy, 2005)
Fruchterman-Reingold Layout: Gold standard for network visualization, provides aesthetically pleasing and interpretable layouts (Fruchterman & Reingold, 1991)
Relevance Weighting: Incorporates the semantic importance of concepts beyond mere frequency, improving network quality
Temporal Analysis: Reveals evolution of research themes over time, critical for understanding field development

Parameters to consider adjusting:
min_concept_frequency: Increase for cleaner networks, decrease for comprehensive coverage
relevance_threshold: Higher values focus on more relevant concepts
top_concepts_viz: Adjust based on network size and visualization clarity needs
"""

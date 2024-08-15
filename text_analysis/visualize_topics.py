import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict
import matplotlib.colors as mcolors

def create_topic_timeline(df, video_id):
    # Create a larger color palette
    colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    fig, ax = plt.subplots(figsize=(20, 10))
    unique_topics = df['topic'].unique()
    color_dict = {topic: mymap(i/len(unique_topics)) for i, topic in enumerate(unique_topics)}
    
    for _, row in df.iterrows():
        color = color_dict[row['topic']]
        ax.barh(row['turn_index'], row['end'] - row['start'], left=row['start'],
                height=0.8, color=color, alpha=0.8)
    
    ax.set_yticks(df['turn_index'])
    ax.set_ylabel('Turn Index', fontsize=14)
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_title(f'Topic Timeline for {video_id}', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color_dict[t]) for t in unique_topics]
    labels = [topic_dict.get(t, f'Topic {t}') for t in unique_topics]
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{BASE_PATH}plots/topic_timeline_{video_id}.png', bbox_inches='tight')
    plt.close()


def create_topic_heatmap(df, video_id):
    df['time_bin'] = pd.cut(df['start'], bins=range(0, int(df['end'].max()) + 60, 60))
    pivot = pd.pivot_table(df, values='end', index='topic', columns='time_bin',
                           aggfunc='count', fill_value=0)
    
    # Replace topic numbers with names
    pivot.index = pivot.index.map(lambda x: topic_dict.get(x, f'Topic {x}'))
    
    plt.figure(figsize=(20, 15))
    sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Frequency'})
    plt.title(f'Topic Heatmap for {video_id}', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=16)
    plt.ylabel('Topic', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{BASE_PATH}plots/topic_heatmap_{video_id}.png', bbox_inches='tight')
    plt.close()


def create_speaker_turn_network(df, video_id):
    G = nx.DiGraph()
    for i in range(len(df) - 1):
        source = df.iloc[i]['topic']
        target = df.iloc[i+1]['topic']
        if G.has_edge(source, target):
            G[source][target]['weight'] += 1
        else:
            G.add_edge(source, target, weight=1)
    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=[G.degree(n) * 100 for n in G.nodes()],
                           node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                           width=[G[u][v]['weight'] for u,v in G.edges()])
    
    # Use topic names for labels
    labels = {node: topic_dict.get(node, f'Topic {node}') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=14)
    
    plt.title(f'Speaker Turn Network Graph for {video_id}', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{BASE_PATH}plots/speaker_turn_network_{video_id}.png', bbox_inches='tight')
    plt.close()

def merge_consecutive_topics(df):
    merged_data = []
    current_segment = None
    for _, row in df.iterrows():
        if current_segment is None:
            current_segment = row.to_dict()
        elif row['topic'] == current_segment['topic']:
            current_segment['end'] = row['end']
        else:
            merged_data.append(current_segment)
            current_segment = row.to_dict()
    if current_segment is not None:
        merged_data.append(current_segment)
    return pd.DataFrame(merged_data)

def create_visualizations_for_videos(num_videos):
    unique_videos = speaker_turns_df['video'].unique()
    videos_to_process = unique_videos[:num_videos]
    for video_id in videos_to_process:
        video_data = speaker_turns_df[speaker_turns_df['video'] == video_id]
        # Merge consecutive topics
        merged_video_data = merge_consecutive_topics(video_data)
        create_topic_timeline(merged_video_data, video_id)
        create_topic_heatmap(merged_video_data, video_id)
        create_speaker_turn_network(merged_video_data, video_id)
        print(f"Created visualizations for {video_id}")



# Set the base path as a variable
BASE_PATH = "text_analysis/topics/welt/"
# Load the data
speaker_turns_df = pd.read_csv(f'{BASE_PATH}speaker_turn_topics.csv')
topic_info_df = pd.read_csv(f'{BASE_PATH}turn_topic_info.csv')

topic_dict = dict(zip(topic_info_df['Topic'], topic_info_df['Name']))

create_visualizations_for_videos(5)
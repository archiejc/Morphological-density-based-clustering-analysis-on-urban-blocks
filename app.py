
import pandas as pd
import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import umap.umap_ as umap
import hdbscan
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist
import json
import os
import socket
import warnings
warnings.filterwarnings('ignore')

class DeployableHierarchicalClustering:
    def __init__(self, nodes_df, embeddings, shapefile_path=None, max_samples_for_dendrogram=200):
        self.nodes_df = nodes_df
        self.embeddings = embeddings
        self.shapefile_path = shapefile_path
        self.max_samples_for_dendrogram = max_samples_for_dendrogram
        
        # Data processing results
        self.umap_embedding = None
        self.clusterer = None
        self.gdf = None
        self.shape_gdf = None
        
        # Hierarchy related
        self.scipy_linkage = None
        self.dendrogram_sample_indices = None
        self.height_min = 0.01
        self.height_max = 2.0
        
        # Pre-computed clustering results at different heights
        self.precomputed_labels = {}
        self.height_levels = None
        
        # Color mapping
        self.color_palette = px.colors.qualitative.Set3
        
        print("🚀 Initializing deployable clustering app...")
        # Initialize data
        self._setup_data()
        
        # Setup fixed display bounds
        self._setup_fixed_display_bounds()
        
        # Create Dash app
        self.app = dash.Dash(__name__)
        self.app.title = "Interactive Hierarchical Clustering - Nanjing Street Network"
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_data(self):
        """Initialize data processing with optimizations"""
        print("📊 Performing UMAP dimensionality reduction...")
        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(self.embeddings)//10),
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            n_jobs=1
        )
        self.umap_embedding = umap_reducer.fit_transform(self.embeddings)
        
        print("🔍 Performing HDBSCAN clustering...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, len(self.embeddings)//50),
            min_samples=3,
            metric='euclidean',
            cluster_selection_method='eom',
            core_dist_n_jobs=1
        )
        self.clusterer.fit(self.umap_embedding)
        
        # Create geographic dataframe
        geometry = [Point(xy) for xy in zip(self.nodes_df['Longitude'], self.nodes_df['Latitude'])]
        self.gdf = gpd.GeoDataFrame(self.nodes_df.copy(), geometry=geometry)
        self.gdf.crs = 'EPSG:4326'
        
        # Load shapefile (optional)
        self._load_shapefile()
        
        # Build lightweight hierarchy
        self._build_lightweight_hierarchy()
        
        print(f"✅ Setup complete - UMAP: {self.umap_embedding.shape}")
        print(f"📏 Height range: {self.height_min:.4f} -> {self.height_max:.4f}")
    
    def _setup_fixed_display_bounds(self):
        """Setup fixed boundaries for both map and UMAP to maintain consistent aspect ratio"""
        
        # 🗺️ 地图固定边界
        lat_min, lat_max = self.nodes_df['Latitude'].min(), self.nodes_df['Latitude'].max()
        lon_min, lon_max = self.nodes_df['Longitude'].min(), self.nodes_df['Longitude'].max()
        
        lat_buffer = (lat_max - lat_min) * 0.05
        lon_buffer = (lon_max - lon_min) * 0.05
        
        # 🎯 UMAP固定边界
        umap_x_min, umap_x_max = self.umap_embedding[:, 0].min(), self.umap_embedding[:, 0].max()
        umap_y_min, umap_y_max = self.umap_embedding[:, 1].min(), self.umap_embedding[:, 1].max()
        
        umap_x_buffer = (umap_x_max - umap_x_min) * 0.05
        umap_y_buffer = (umap_y_max - umap_y_min) * 0.05
        
        # 保存所有固定配置
        self.fixed_display_config = {
            'map': {
                'center_lat': (lat_max + lat_min) / 2,
                'center_lon': (lon_max + lon_min) / 2,
                'bounds': {
                    'west': lon_min - lon_buffer,
                    'east': lon_max + lon_buffer,
                    'south': lat_min - lat_buffer,
                    'north': lat_max + lat_buffer
                },
                'zoom': 11
            },
            'umap': {
                'x_range': [umap_x_min - umap_x_buffer, umap_x_max + umap_x_buffer],
                'y_range': [umap_y_min - umap_y_buffer, umap_y_max + umap_y_buffer],
                'aspect_ratio': 'equal'
            }
        }
        
        print(f"🔒 Fixed map bounds: Lat {lat_min:.4f}-{lat_max:.4f}, Lon {lon_min:.4f}-{lon_max:.4f}")
        print(f"🔒 Fixed UMAP bounds: X {umap_x_min:.4f}-{umap_x_max:.4f}, Y {umap_y_min:.4f}-{umap_y_max:.4f}")
    
    def _load_shapefile(self):
        """Load shapefile with error handling"""
        if self.shapefile_path and os.path.exists(self.shapefile_path):
            try:
                self.shape_gdf = gpd.read_file(self.shapefile_path)
                if self.shape_gdf.crs != self.gdf.crs:
                    self.shape_gdf = self.shape_gdf.to_crs(self.gdf.crs)
                
                joined = gpd.sjoin(self.gdf, self.shape_gdf, how='left', predicate='within')
                self.gdf = joined
                print(f"✅ Successfully loaded shapefile: {self.shape_gdf.shape[0]} blocks")
            except Exception as e:
                print(f"⚠️ Failed to load shapefile: {e}")
                self.shape_gdf = None
        else:
            print("📍 No shapefile provided, using point display")
            self.shape_gdf = None
    
    def _build_lightweight_hierarchy(self):
        """Build lightweight hierarchy with sampling"""
        print("🌳 Building lightweight hierarchy...")
        
        n_samples = len(self.umap_embedding)
        
        if n_samples > self.max_samples_for_dendrogram:
            print(f"📊 Sampling {self.max_samples_for_dendrogram} points for dendrogram from {n_samples} total")
            labels = self.clusterer.labels_
            self.dendrogram_sample_indices = []
            
            unique_labels = np.unique(labels)
            samples_per_cluster = self.max_samples_for_dendrogram // len(unique_labels)
            
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                n_to_sample = min(samples_per_cluster, len(cluster_indices))
                sampled = np.random.choice(cluster_indices, n_to_sample, replace=False)
                self.dendrogram_sample_indices.extend(sampled)
            
            remaining = self.max_samples_for_dendrogram - len(self.dendrogram_sample_indices)
            if remaining > 0:
                all_indices = set(range(n_samples))
                unused_indices = list(all_indices - set(self.dendrogram_sample_indices))
                if len(unused_indices) >= remaining:
                    additional = np.random.choice(unused_indices, remaining, replace=False)
                    self.dendrogram_sample_indices.extend(additional)
            
            self.dendrogram_sample_indices = np.array(self.dendrogram_sample_indices)
            sample_data = self.umap_embedding[self.dendrogram_sample_indices]
        else:
            self.dendrogram_sample_indices = np.arange(n_samples)
            sample_data = self.umap_embedding
        
        try:
            print("🔄 Computing pairwise distances...")
            distances = pdist(sample_data, metric='euclidean')
            print("🔗 Performing hierarchical clustering...")
            self.scipy_linkage = linkage(distances, method='single')
            
            heights = self.scipy_linkage[:, 2]
            self.height_min = max(heights.min(), 0.001)
            self.height_max = heights.max()
            
            if self.height_max - self.height_min < 0.01:
                self.height_max = self.height_min + 0.5
                
            print(f"✅ Hierarchy built successfully")
            
        except Exception as e:
            print(f"❌ Error in hierarchy building: {e}")
            self.height_min = 0.01
            self.height_max = 2.0
            self.scipy_linkage = None
        
        self._precompute_clusters()
    
    def _precompute_clusters(self):
        """Pre-compute clustering results at several heights"""
        print("⚡ Pre-computing cluster assignments...")
        
        self.height_levels = np.linspace(self.height_min, self.height_max, 20)
        
        for height in self.height_levels:
            labels = self._compute_clusters_at_height(height)
            self.precomputed_labels[height] = labels
        
        print(f"✅ Pre-computed clusters for {len(self.height_levels)} height levels")
    
    def _compute_clusters_at_height(self, height):
        """Compute cluster labels at specific height"""
        try:
            if self.scipy_linkage is not None:
                sample_clusters = cut_tree(self.scipy_linkage, height=height).flatten()
                
                if len(self.dendrogram_sample_indices) < len(self.umap_embedding):
                    from sklearn.cluster import KMeans
                    
                    unique_sample_labels = np.unique(sample_clusters)
                    centers = []
                    for label in unique_sample_labels:
                        mask = sample_clusters == label
                        if np.sum(mask) > 0:
                            center = self.umap_embedding[self.dendrogram_sample_indices[mask]].mean(axis=0)
                            centers.append(center)
                    
                    if len(centers) > 0:
                        centers = np.array(centers)
                        from sklearn.metrics.pairwise import euclidean_distances
                        distances = euclidean_distances(self.umap_embedding, centers)
                        full_labels = np.argmin(distances, axis=1)
                    else:
                        full_labels = np.zeros(len(self.umap_embedding))
                else:
                    full_labels = sample_clusters
                
                return full_labels
            else:
                return self.clusterer.labels_
                
        except Exception as e:
            print(f"❌ Error computing clusters at height {height}: {e}")
            return self.clusterer.labels_
    
    def _get_clusters_at_height(self, height):
        """Get clusters at specified height with interpolation"""
        if len(self.precomputed_labels) == 0:
            return self.clusterer.labels_
        
        closest_height = min(self.height_levels, key=lambda x: abs(x - height))
        
        if abs(closest_height - height) < (self.height_max - self.height_min) / 100:
            return self.precomputed_labels[closest_height]
        else:
            return self._compute_clusters_at_height(height)
    
    def _setup_layout(self):
        """Setup Dash layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🌐 Interactive Hierarchical Clustering", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
                html.H2("Nanjing Street Network Analysis", 
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30, 'fontSize': 20})
            ]),
            
            # Loading indicator
            dcc.Loading(
                id="loading",
                type="default",
                children=[
                    # Control panel
                    html.Div([
                        html.Div([
                            html.Label("🎛️ Hierarchy Cut Control (Leaf to Root):", 
                                     style={'fontWeight': 'bold', 'fontSize': 16, 'color': '#2c3e50'}),
                            dcc.Slider(
                                id='height-slider',
                                min=self.height_min,
                                max=self.height_max,
                                value=self.height_min + (self.height_max - self.height_min) * 0.3,
                                marks={
                                    self.height_min: f'🍃 Leaf {self.height_min:.3f}',
                                    self.height_max: f'🌳 Root {self.height_max:.3f}'
                                },
                                tooltip={"placement": "bottom", "always_visible": True},
                                step=(self.height_max - self.height_min) / 100
                            )
                        ], style={'width': '70%', 'display': 'inline-block', 'paddingRight': '20px'}),
                        
                        html.Div([
                            html.Label("🎯 Precise Input:", 
                                     style={'fontWeight': 'bold', 'fontSize': 14, 'color': '#2c3e50'}),
                            dcc.Input(
                                id='height-input',
                                type='number',
                                value=self.height_min + (self.height_max - self.height_min) * 0.3,
                                min=self.height_min,
                                max=self.height_max,
                                step=(self.height_max - self.height_min) / 100,
                                style={'width': '100%', 'padding': '5px', 'borderRadius': '5px', 'border': '1px solid #bdc3c7'}
                            )
                        ], style={'width': '25%', 'display': 'inline-block', 'marginLeft': '5%'})
                    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px'}),
                    
                    # Info panel
                    html.Div([
                        html.P([
                            "📊 Dendrogram computed on ",
                            html.Strong(f"{len(self.dendrogram_sample_indices)} samples"),
                            f" (sampled from {len(self.umap_embedding)} total)" if len(self.dendrogram_sample_indices) < len(self.umap_embedding) else "",
                            " | 🔒 Fixed aspect ratios for consistent visualization"
                        ], style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#7f8c8d'})
                    ]),
                    
                    # Dendrogram
                    html.Div([
                        dcc.Graph(
                            id='dendrogram-plot',
                            style={'height': '400px'},
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], style={'margin': '10px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    
                    # Main visualization area
                    html.Div([
                        # UMAP scatter plot
                        html.Div([
                            dcc.Graph(
                                id='umap-plot',
                                style={'height': '450px'},
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ], style={'width': '33%', 'display': 'inline-block', 'padding': '5px'}),
                        
                        # Geographic plot
                        html.Div([
                            dcc.Graph(
                                id='geo-plot',
                                style={'height': '450px'},
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ], style={'width': '34%', 'display': 'inline-block', 'padding': '5px'}),
                        
                        # Statistics plot
                        html.Div([
                            dcc.Graph(
                                id='stats-plot',
                                style={'height': '450px'},
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ], style={'width': '33%', 'display': 'inline-block', 'padding': '5px'})
                    ], style={'backgroundColor': 'white', 'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    
                    # Information panel
                    html.Div([
                        html.Div(id='cluster-info', 
                                style={'backgroundColor': '#f8f9fa', 'padding': '20px', 
                                      'borderRadius': '10px', 'margin': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                    ])
                ]
            ),
            
            # Footer
            html.Div([
                html.Hr(),
                html.P([
                    "🔬 Built with ",
                    html.A("Dash", href="https://dash.plotly.com/", target="_blank"),
                    " | 📊 Data Analysis & Visualization | 🚀 Deployed on the Web"
                ], style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 12})
            ])
        ])
    
    def _setup_callbacks(self):
        """Setup callback functions"""
        @self.app.callback(
            [Output('height-slider', 'value'),
             Output('height-input', 'value')],
            [Input('height-slider', 'value'),
             Input('height-input', 'value'),
             Input('dendrogram-plot', 'clickData')],
            [State('height-slider', 'value'),
             State('height-input', 'value')]
        )
        def sync_height_controls(slider_val, input_val, click_data, current_slider, current_input):
            """Synchronize height value controls"""
            ctx = callback_context
            if not ctx.triggered:
                return (self.height_min + (self.height_max - self.height_min) * 0.3, 
                       self.height_min + (self.height_max - self.height_min) * 0.3)
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'dendrogram-plot' and click_data:
                if 'y' in click_data['points'][0]:
                    new_height = click_data['points'][0]['y']
                    new_height = np.clip(new_height, self.height_min, self.height_max)
                    return new_height, new_height
            elif trigger_id == 'height-slider':
                return slider_val, slider_val
            elif trigger_id == 'height-input':
                clamped_val = np.clip(input_val or self.height_min, self.height_min, self.height_max)
                return clamped_val, clamped_val
            
            return current_slider, current_input
        
        @self.app.callback(
            [Output('dendrogram-plot', 'figure'),
             Output('umap-plot', 'figure'),
             Output('geo-plot', 'figure'),
             Output('stats-plot', 'figure'),
             Output('cluster-info', 'children')],
            [Input('height-slider', 'value')]
        )
        def update_all_plots(height_val):
            """Update all plots"""
            try:
                labels = self._get_clusters_at_height(height_val)
                
                dendrogram_fig = self._create_dendrogram_figure(height_val)
                umap_fig = self._create_umap_figure(labels, height_val)
                geo_fig = self._create_geo_figure(labels, height_val)
                stats_fig = self._create_stats_figure(labels, height_val)
                info_div = self._create_info_panel(labels, height_val)
                
                return dendrogram_fig, umap_fig, geo_fig, stats_fig, info_div
            except Exception as e:
                print(f"❌ Error in callback: {e}")
                empty_fig = go.Figure()
                return empty_fig, empty_fig, empty_fig, empty_fig, [html.P(f"Error: {str(e)}")]
    
    def _create_dendrogram_figure(self, current_height):
        """Create dendrogram figure"""
        try:
            if self.scipy_linkage is not None:
                fig = go.Figure()
                
                from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
                
                ddata = scipy_dendrogram(
                    self.scipy_linkage,
                    no_plot=True,
                    count_sort=True,
                    distance_sort=True
                )
                
                for i, d in enumerate(ddata['dcoord']):
                    fig.add_trace(go.Scatter(
                        x=ddata['icoord'][i],
                        y=d,
                        mode='lines',
                        line=dict(color='#34495e', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                fig.add_hline(
                    y=current_height,
                    line_dash="dash",
                    line_color="#e74c3c",
                    line_width=3,
                    annotation_text=f"✂️ Cut: {current_height:.3f}"
                )
                
                fig.update_layout(
                    title=f"🌳 Dendrogram (n={len(self.dendrogram_sample_indices)})",
                    xaxis_title="Sample Index",
                    yaxis_title="Distance",
                    hovermode='closest',
                    height=400,
                    xaxis=dict(showticklabels=False),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                return fig
            else:
                return self._create_simple_tree_figure(current_height)
                
        except Exception as e:
            print(f"❌ Error creating dendrogram: {e}")
            return self._create_simple_tree_figure(current_height)
    
    def _create_simple_tree_figure(self, current_height):
        """Create simple tree representation"""
        fig = go.Figure()
        
        n_levels = 5
        heights = np.linspace(self.height_min, self.height_max, n_levels)
        
        for i, h in enumerate(heights):
            fig.add_hline(
                y=h,
                line=dict(color='#bdc3c7', width=1),
                opacity=0.5
            )
        
        fig.add_hline(
            y=current_height,
            line_dash="dash",
            line_color="#e74c3c",
            line_width=3,
            annotation_text=f"✂️ Cut Height: {current_height:.3f}"
        )
        
        fig.update_layout(
            title="🌳 Hierarchy Levels",
            xaxis_title="Structure",
            yaxis_title="Distance",
            height=400,
            xaxis=dict(range=[0, 10], showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def _create_umap_figure(self, labels, height):
        """Create UMAP scatter plot with FIXED aspect ratio and range"""
        unique_labels = np.unique(labels)
        fig = go.Figure()
        
        colors = self.color_palette * (len(unique_labels) // len(self.color_palette) + 1)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = '#95a5a6' if label == -1 else colors[i % len(colors)]
            name = '🔇 Noise' if label == -1 else f'🎯 Cluster {label}'
            
            fig.add_trace(go.Scatter(
                x=self.umap_embedding[mask, 0],
                y=self.umap_embedding[mask, 1],
                mode='markers',
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                name=name,
                text=[f"Node: {idx}<br>Cluster: {label}" for idx in np.where(mask)[0]],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # 🔑 使用固定的UMAP显示配置
        fig.update_layout(
            title=f"🎯 UMAP Embedding (h={height:.3f})",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            showlegend=len(unique_labels) <= 10,
            height=450,
            xaxis=dict(
                range=self.fixed_display_config['umap']['x_range'],
                constrain='domain',
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=self.fixed_display_config['umap']['y_range'],
                constrain='domain'
            ),
            autosize=False,
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def _create_geo_figure(self, labels, height):
        """Create geographic plot with FIXED aspect ratio"""
        unique_labels = np.unique(labels)
        fig = go.Figure()
        
        colors = self.color_palette * (len(unique_labels) // len(self.color_palette) + 1)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = '#95a5a6' if label == -1 else colors[i % len(colors)]
            name = '🔇 Noise' if label == -1 else f'🎯 Cluster {label}'
            
            fig.add_trace(go.Scattermapbox(
                lat=self.nodes_df.loc[mask, 'Latitude'],
                lon=self.nodes_df.loc[mask, 'Longitude'],
                mode='markers',
                marker=dict(color=color, size=8, opacity=0.7),
                name=name,
                text=[f"Node: {idx}<br>Cluster: {label}" for idx in np.where(mask)[0]],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # 🔑 使用固定的地图配置
        map_config = self.fixed_display_config['map']
        
        fig.update_layout(
            title=f"🗺️ Geographic Distribution (h={height:.3f})",
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=map_config['center_lat'], 
                    lon=map_config['center_lon']
                ),
                zoom=map_config['zoom'],
                bounds=map_config['bounds']
            ),
            showlegend=len(unique_labels) <= 10,
            height=450,
            autosize=False,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def _create_stats_figure(self, labels, height):
        """Create statistics plot"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(labels == -1)
        
        if n_clusters > 0:
            cluster_sizes = []
            cluster_names = []
            
            for label in unique_labels:
                if label != -1:
                    size = np.sum(labels == label)
                    cluster_sizes.append(size)
                    cluster_names.append(f'C{label}')
            
            colors_list = self.color_palette * (len(cluster_sizes) // len(self.color_palette) + 1)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=cluster_names,
                    y=cluster_sizes,
                    marker_color=colors_list[:len(cluster_sizes)],
                    text=cluster_sizes,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"📊 Clusters: {n_clusters}, Noise: {n_noise}",
                xaxis_title="Clusters",
                yaxis_title="Size (# nodes)",
                height=450,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
        else:
            fig = go.Figure()
            fig.add_annotation(
                text=f"🚫 No Clusters<br>{n_noise} noise points",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color='#7f8c8d')
            )
            fig.update_layout(
                height=450,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
        
        return fig
    
    def _create_info_panel(self, labels, height):
        """Create information panel"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(labels == -1)
        
        position_ratio = (height - self.height_min) / (self.height_max - self.height_min) * 100
        
        info_items = [
            html.H4(f"📊 Current Analysis", style={'color': '#2c3e50'}),
            html.P([
                html.Strong("Cut Height: "), f"{height:.4f}",
                html.Br(),
                html.Strong("Position: "), f"{position_ratio:.1f}% (Leaf→Root)",
                html.Br(),
                html.Strong("Clusters: "), f"{n_clusters}",
                html.Br(),
                html.Strong("Noise Points: "), f"{n_noise}",
                html.Br(),
                html.Strong("Height Range: "), f"{self.height_min:.3f} → {self.height_max:.3f}"
            ]),
        ]
        
        if n_clusters > 0 and n_clusters <= 20:
            info_items.append(html.Hr())
            info_items.append(html.H5("🎯 Cluster Details", style={'color': '#2c3e50'}))
            for label in unique_labels:
                if label != -1:
                    size = np.sum(labels == label)
                    percentage = size / len(labels) * 100
                    info_items.append(
                        html.P(f"Cluster {label}: {size} nodes ({percentage:.1f}%)", 
                              style={'margin': '5px 0'})
                    )
        
        return info_items
    
    def get_app(self):
        """Get the Dash app for deployment"""
        return self.app

def load_data():
    """Load data"""
    try:
        nodes_df = pd.read_csv('hdbscan_cluster_results.csv')
        embeddings = torch.load('nanjing_embeddings_for_clustering(256).pt', 
                               map_location=torch.device('cpu'))
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        
        print(f"✅ Data loaded: {len(nodes_df)} nodes, {embeddings.shape} embeddings")
        return nodes_df, embeddings
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def create_app():
    """Create and return the Dash app"""
    print("🚀 === Deployable Interactive Hierarchical Clustering ===")
    
    # Load data
    nodes_df, embeddings = load_data()
    
    # Check for shapefile
    shapefile_path = None
    common_names = ['nanjing_blocks.shp', 'blocks.shp', 'districts.shp', 'polygons.shp']
    for name in common_names:
        if os.path.exists(name):
            shapefile_path = name
            break
    
    # Create application
    clustering_app = DeployableHierarchicalClustering(
        nodes_df, embeddings, 
        shapefile_path=shapefile_path,
        max_samples_for_dendrogram=200
    )
    
    return clustering_app.get_app()

# For deployment
app = create_app()
server = app.server

if __name__ == "__main__":
    # For local development
    app.run_server(debug=True, host='0.0.0.0', port=8050)

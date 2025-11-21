
"""
Client API interactif pour interroger la base de données Twopercenters 
et générer des visualisations Plotly avec widgets ipywidgets.

"""

import requests
import base64
import zlib
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import country_converter as coco
from typing import Dict, List, Optional, Any, Tuple
from IPython.display import display, clear_output,  HTML
import ipywidgets as widgets
import dash_daq as daq


class TwoPercentersClient:
    """Client pour interroger l'API Twopercenters et créer des visualisations."""
    
    def __init__(self, base_url: str = "https://twopercenters.db.neurolibre.org/api/v1"):
        """
        Initialise le client API.
        
        Args:
            base_url: URL de base de l'API
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Couleurs du dashboard
        self.darkAccent1 = '#2C2C2C'
        self.darkAccent2 = '#5b5959'
        self.darkAccent3 = '#CFCFCF'
        self.lightAccent1 = '#ECAB4C'
        self.highlight1 = 'lightsteelblue'
        self.highlight2 = 'cornflowerblue'
        self.bgc = self.darkAccent1
        
        # Cache pour les recherches
        self._author_cache = {}
        
    # ========================================================================
    # MÉTHODES UTILITAIRES
    # ========================================================================
    
    def _get_metric_long_name(self, career: bool, metric: str) -> str:
        """Obtient le nom complet d'une métrique."""
        metric_names = {
            'rank (ns)': 'Rank (no self-citations)',
            'nc (ns)': 'Total citations (no self-cit.)',
            'h (ns)': 'H-index (no self-cit.)',
            'hm (ns)': 'Hm-index (no self-cit.)',
            'nps (ns)': '# single authored papers (no self-cit.)',
            'ncs (ns)': 'Cit. to single authored papers (no self-cit.)',
            'cpsf (ns)': '# single + first authored papers (no self-cit.)',
            'ncsf (ns)': 'Cit. to single + first authored (no self-cit.)',
            'npsfl (ns)': '# single + first + last authored (no self-cit.)',
            'ncsfl (ns)': 'Cit. to single + first + last (no self-cit.)',
            'c (ns)': 'Composite score (no self-cit.)',
            'npciting (ns)': '# distinct citing papers (no self-cit.)',
            'rank': 'Rank',
            'nc': 'Total citations',
            'h': 'H-index',
            'hm': 'Hm-index',
            'nps': '# single authored papers',
            'ncs': 'Citations to single authored papers',
            'cpsf': '# single + first authored papers',
            'ncsf': 'Citations to single + first authored',
            'npsfl': '# single + first + last authored',
            'ncsfl': 'Citations to single + first + last',
            'c': 'Composite score',
            'npciting': '# distinct citing papers',
            'np': '# papers',
            'self%': 'Self-citation %'
        }
        return metric_names.get(metric, metric)
    
    #utils coming from twopercenters dashboard 
    def get_aggregate_data(self, group, group_name, prefix):
        """
        Récupère les données agrégées pour un groupe donné (pays, domaine, institution)
        Basé sur la fonction get_es_aggregate du serveur
        """
        try:
            if group == 'cntry':
                # Convertir le nom du pays en code ISO3 comme dans le serveur
                import country_converter as coco
                cur_country = coco.convert(names=group_name, to='ISO3')
                results = self.get_es_results(cur_country.lower(), f'{prefix}_cntry', 'cntry', True)
            elif group == "sm-field":
                results = self.get_es_results(group_name, f'{prefix}_field', 'sm-field')
            elif group == "inst_name":
                results = self.get_es_results(group_name, f'{prefix}_inst', 'inst_name')
            else:
                return None
                
            if results is not None:
                data = self.es_result_pick(results, 'data', None)
                return data
            else:
                return None
                
        except Exception as e:
            print(f"Erreur dans get_aggregate_data: {e}")
            return None

    def get_es_results(self, search_term, idx_name, search_fields, exact=False):
        """
        Recherche dans ElasticSearch - version adaptée pour le client
        """
        # Vous devrez adapter cette fonction selon votre configuration client
        # Pour l'instant, je vais utiliser une version simplifiée
        try:
            # Si vous avez accès à l'instance es dans votre client:
            if exact:
                result = self.es.search(
                    index=idx_name,
                    size=100,
                    body={
                        "query": {
                            "term": {
                                search_fields: search_term
                            },
                        }
                    })
            else:
                result = self.es.search(
                    index=idx_name,
                    size=100, 
                    body={
                        "query": {
                            "multi_match": {
                                "query": search_term, 
                                "operator": "and",
                                "fuzziness": "auto",
                                "fields": [search_fields]
                            },
                        }
                    })
                    
            if 'hits' in result and 'hits' in result['hits'] and result['hits']['hits']:
                return pd.json_normalize(result['hits']['hits'])
            else:
                return None
                
        except Exception as e:
            print(f"Erreur ES: {e}")
            return None

    def es_result_pick(self, result, field, nohit=['']):
        """
        Extrait les données des résultats ES - version client
        """
        if result is not None:
            if field == 'data':
                return self.base64_decode_and_decompress(result[f'_source.{field}'])
            elif (f'_source.{field}' in result.keys()):
                return list(result[f'_source.{field}'])
            else:
                return nohit
        else:
            return nohit

    def base64_decode_and_decompress(self, encoded_data, flg=True):
        """
        Décode les données compressées - version client
        """

        # if flg:
        #     encoded_data = encoded_data[0]

        # Base64 decode the data
        compressed_data = base64.b64decode(encoded_data)
        
        # Decompress the data using zlib
        decompressed_data = zlib.decompress(compressed_data)
        
        # Convert the decompressed string back to a dictionary
        decoded_data = json.loads(decompressed_data.decode('utf-8'))
        
        return decoded_data

    # ========================================================================
    # MÉTHODES D'INTERROGATION API
    # ========================================================================
    
    def search_authors(self, query: str, index: str = "career", 
                      field: str = "authfull", limit: int = 10) -> List[Dict]:
        """Recherche des auteurs dans la base de données."""
        url = f"{self.base_url}/search/authors"
        payload = {
            "query": query,
            "index": index,
            "field": field,
            "limit": limit
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Décompresser les données si présentes
        if "results" in data:
            for result in data["results"]:
                if "data" in result:
                    result["data"] = self.base64_decode_and_decompress(result["data"])
        
        return data["results"]
    
    def get_author_data(self, author_name: str, index: str = "career") -> Optional[Dict]:
        """Récupère les données complètes d'un auteur."""
        cache_key = f"{author_name}_{index}"
        if cache_key in self._author_cache:
            return self._author_cache[cache_key]
        
        results = self.search_authors(author_name, index=index, field="authfull", limit=5)
        
        if not results:
            return None
        
        # Chercher la correspondance exacte
        for result in results:
            if result.get("authfull") == author_name:
                data = result.get("data")
                self._author_cache[cache_key] = data
                return data
        
        # Si pas de correspondance exacte, retourner le premier résultat
        data = results[0].get("data") if results else None
        self._author_cache[cache_key] = data
        return data
    
    def get_author_info(self, author_name: str, index: str = "career") -> Dict[str, Any]:
        """Récupère les informations de base d'un auteur."""
        results = self.search_authors(author_name, index=index, field="authfull", limit=1)
        
        if not results:
            return {}
        
        result = results[0]
        data = result.get("data", {})
        
        # Extraire la dernière année disponible
        years = [k.split('_')[1] for k in data.keys() if not k.endswith('_log')]
        if years:
            latest_year = max(years)
            latest_data = data.get(f"{index}_{latest_year}", {})
            
            return {
                "author": author_name,
                "institution": latest_data.get("inst_name", "N/A"),
                "country": latest_data.get("cntry", "N/A"),
                "field": latest_data.get("sm-field", "N/A"),
                "self_citation_pct": round(latest_data.get("self%", 0) * 100, 2),
                "rank": latest_data.get("rank", "N/A"),
                "h_index": latest_data.get("h", "N/A"),
                "total_citations": latest_data.get("nc", "N/A"),
                "latest_year": latest_year
            }
        
        return {}
    
    # ========================================================================
    # VISUALISATIONS - AUTEUR INDIVIDUEL
    # ========================================================================
    
    def plot_author_metrics_gauge(self, author_name: str, year: str = "2021",
                                  career: bool = True, exclude_self_citations: bool = False,
                                  comparison_group: str = "country") -> go.Figure:
        """
        Crée des jauges comparant les métriques d'un auteur aux médianes/max du groupe.
        Reproduit la visualisation du dashboard (author_find.py).
        
        Args:
            author_name: Nom de l'auteur
            year: Année à afficher
            career: True pour career, False pour single year
            exclude_self_citations: Exclure les auto-citations
            comparison_group: "country", "field", ou "institution"
            
        Returns:
            Figure Plotly avec gauges
        """
        index = "career" if career else "singleyr"
        prefix = "career" if career else "singleyr"
        
        # Récupérer les données de l'auteur
        data = self.get_author_data(author_name, index=index)
        
        if not data:
            print(f"Aucune donnée trouvée pour {author_name}")
            return go.Figure()
        
        key = f"{prefix}_{year}"
        if key not in data:
            print(f"Données non disponibles pour l'année {year}")
            return go.Figure()
        
        author_data = data[key]
        
        # Récupérer les données agrégées pour comparaison
        info = self.get_author_info(author_name, index=index)
        
        # Préparer les métriques
        suffix = ' (ns)' if exclude_self_citations else ''
        metrics_list = [f'nc{suffix}', f'h{suffix}', f'hm{suffix}', 
                       f'ncs{suffix}', f'ncsf{suffix}', f'ncsfl{suffix}']
        
        metric_titles = ['Total Citations', 'H-index', 'Hm-index',
                        'Cit. Single Auth.', 'Cit. Single+First', 'Cit. Single+First+Last']
        
        # Créer la figure avec sous-graphiques
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'indicator'}] * 3,
                   [{'type': 'indicator'}] * 3],
            subplot_titles=metric_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Valeurs de l'auteur
        author_values = [author_data.get(m, 0) for m in metrics_list]
        
        # Dans la vraie implémentation, ces valeurs viendraient de get_es_aggregate
        for i, (metric, title, value) in enumerate(zip(metrics_list, metric_titles, author_values)):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            # Estimer médiane et max (à remplacer par vraies valeurs API)
            median = value * 0.7 if value > 0 else 100
            max_val = value * 1.5 if value > 0 else 1000
            
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
                    delta={'reference': median, 
                          'increasing': {'color': "limegreen"},
                          'decreasing': {'color': "indianred"}},
                    gauge={
                        'axis': {'range': [None, max_val], 'tickcolor': "#aaa"},
                        'bar': {'color': 'lightseagreen'},
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': median
                        }
                    },
                    title={'text': title, 'font': {'color': '#aaa', 'size': 12}}
                ),
                row=row, col=col
            )
        
        # Mise en forme
        fig.update_layout(
            height=500,
            plot_bgcolor=self.bgc,
            paper_bgcolor=self.bgc,
            font=dict(color='lightseagreen', size=10),
            title={
                'text': f"{author_name} - Métriques {year}",
                'font': {'size': 16, 'color': self.lightAccent1},
                'x': 0.5
            },
            margin={'l': 20, 'r': 20, 't': 80, 'b': 20}
        )
        
        return fig
    
    def plot_author_career(self, author_name: str, exclude_self_citations: bool = False,
                          width: int = 1200, height: int = 2000) -> go.Figure:
        """Crée un graphique montrant l'évolution des métriques d'un auteur sur sa carrière."""
        data = self.get_author_data(author_name, index="career")
        
        if not data:
            print(f"Aucune donnée trouvée pour {author_name}")
            return go.Figure()
        
        # Extraire les années et les données
        years = []
        metrics_data = []
        
        for key in sorted(data.keys()):
            if key.endswith('_log'):
                continue
            parts = key.split('_')
            if len(parts) == 2 and parts[0] == 'career':
                years.append(parts[1])
                metrics_data.append(data[key])
        
        if not years:
            print(f"Aucune donnée temporelle trouvée pour {author_name}")
            return go.Figure()
        
        # Créer le DataFrame
        df = pd.DataFrame(metrics_data)
        df['Year'] = years
        df['self%'] = df['self%'] * 100
        
        # Définir les métriques à afficher
        suffix = ' (ns)' if exclude_self_citations else ''
        metrics_list = [
            f'rank{suffix}', f'c{suffix}', f'nc{suffix}', f'h{suffix}', 
            f'hm{suffix}', f'ncs{suffix}', f'ncsf{suffix}', f'ncsfl{suffix}',
            f'nps{suffix}', f'cpsf{suffix}', f'npsfl{suffix}', f'npciting{suffix}',
            'np', 'self%'
        ]
        
        # Filtrer les métriques disponibles
        available_metrics = [m for m in metrics_list if m in df.columns]
        
        # Créer les sous-titres
        subplot_titles = [self._get_metric_long_name(True, m) 
                         for m in available_metrics]
        
        # Créer la figure
        fig = make_subplots(
            rows=len(available_metrics), 
            cols=1, 
            subplot_titles=subplot_titles,
            vertical_spacing=0.02
        )
        
        # Palette de couleurs
        col_list = px.colors.sample_colorscale("turbo", 
                                              [n/(100-1) for n in range(100)])
        
        # Ajouter les traces
        for i, metric in enumerate(available_metrics):
            fig.add_trace(
                go.Scatter(
                    x=df['Year'], 
                    y=df[metric],
                    marker=dict(color=col_list[len(col_list) - i - 25]),
                    name=metric,
                    hovertemplate='Année: %{x}<br>Valeur: %{y}<extra></extra>'
                ), 
                row=i + 1, 
                col=1
            )
        
        # Mise en forme
        fig.update_xaxes(
            tickvals=years, 
            ticktext=years, 
            gridcolor=self.darkAccent2, 
            linecolor=self.darkAccent2, 
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor=self.darkAccent2, 
            linecolor=self.darkAccent2, 
            zeroline=False
        )
        fig.update_layout(
            plot_bgcolor=self.bgc, 
            paper_bgcolor=self.bgc, 
            font=dict(color=self.lightAccent1),
            height=height,
            width=width,
            showlegend=False,
            title={
                'text': f"Données carrière: {author_name}", 
                'font': {'size': 20}
            }, 
            title_x=0.5
        )
        
        return fig
    
    def plot_author_comparison(self, author1: str, author2: str, 
                              year: str = "2021", career: bool = True,
                              exclude_self_citations: bool = False,
                              log_transform: bool = False) -> go.Figure:
        """
        Compare deux auteurs sur leurs métriques avec barres.
        Reproduit la visualisation de author_vs_author_layout.
        """
        index = "career" if career else "singleyr"
        prefix = "career" if career else "singleyr"
        
        # Récupérer les données
        data1 = self.get_author_data(author1, index=index)
        data2 = self.get_author_data(author2, index=index)
        
        if not data1 or not data2:
            print("Impossible de récupérer les données pour l'un des auteurs")
            return go.Figure()
        
        # Extraire les données de l'année
        key = f"{prefix}_{year}"
        if key not in data1 or key not in data2:
            print(f"Données non disponibles pour l'année {year}")
            return go.Figure()
        
        metrics1 = data1[key]
        metrics2 = data2[key]
        
        # Définir les métriques à comparer
        suffix = ' (ns)' if exclude_self_citations else ''
        metrics_list = [f'nc{suffix}', f'h{suffix}', f'hm{suffix}', 
                       f'ncs{suffix}', f'ncsf{suffix}', f'ncsfl{suffix}']
        
        metric_titles = ['Citations', 'H-index', 'Hm-index',
                        'Cit. Single', 'Cit. Single+First', 'Cit. Single+First+Last']
        
        # Créer la figure avec sous-graphiques
        fig = make_subplots(
            rows=1, cols=6,
            subplot_titles=metric_titles,
            horizontal_spacing=0.02
        )
        
        # Ajouter les barres pour chaque métrique
        for i, (metric, title) in enumerate(zip(metrics_list, metric_titles)):
            y1 = metrics1.get(metric, 0)
            y2 = metrics2.get(metric, 0)
            
            # Appliquer log si demandé
            if log_transform and metric not in ['c', 'c (ns)']:
                y1 = np.log1p(y1) if y1 > 0 else 0
                y2 = np.log1p(y2) if y2 > 0 else 0
            
            fig.add_trace(
                go.Bar(
                    x=[author1.split(',')[0]],
                    y=[y1],
                    name=author1.split(',')[0],
                    marker_color=self.highlight1,
                    showlegend=(i == 0),
                    hovertemplate=f'{title}<br>%{{y}}<extra></extra>'
                ),
                row=1, col=i+1
            )
            
            fig.add_trace(
                go.Bar(
                    x=[author2.split(',')[0]],
                    y=[y2],
                    name=author2.split(',')[0],
                    marker_color=self.highlight2,
                    showlegend=(i == 0),
                    hovertemplate=f'{title}<br>%{{y}}<extra></extra>'
                ),
                row=1, col=i+1
            )
        
        # Mise en forme
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(gridcolor=self.darkAccent2, linecolor=self.darkAccent2)
        fig.update_layout(
            height=400,
            plot_bgcolor=self.bgc,
            paper_bgcolor=self.bgc,
            font=dict(color=self.lightAccent1),
            title={
                'text': f"Comparaison {author1.split(',')[0]} vs {author2.split(',')[0]} ({year})",
                'font': {'size': 18},
                'x': 0.5
            },
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    # ========================================================================
    # INTERFACES INTERACTIVES AVEC WIDGETS
    # ========================================================================
 
    def get_real_limits_via_api(self, author_data: Dict, comp_group: str, 
                                prefix: str, year: str, suffix: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Récupère les VRAIES limites (max et median) via l'API /aggregate/.
        
        Args:
            author_data: Données de l'auteur
            comp_group: Type de comparaison ('Max and median (red) by country/field/institute')
            prefix: 'career' ou 'singleyr'
            year: Année
            suffix: ' (ns)' ou ''
            
        Returns:
            Tuple (max_limits dict, median_values dict) ou (None, None)
        """
        try:
            # Déterminer le type d'agrégation et la valeur du groupe
            if comp_group == 'Max and median (red) by country':
                api_type = 'country'
                group_value = author_data.get('cntry', '')
            elif comp_group == 'Max and median (red) by field':
                api_type = 'field' 
                group_value = author_data.get('sm-field', '')
            elif comp_group == 'Max and median (red) by institute':
                api_type = 'institution'
                group_value = author_data.get('inst_name', '')
            else:
                return None, None
            
            if not group_value or group_value == 'N/A':
                print(f"⚠️ Valeur de groupe manquante pour {api_type}")
                return None, None
            
            # Construire la requête API
            url = f"{self.base_url}/aggregate/{api_type}"
            params = {'limit': 500}
            
            if prefix == 'singleyr':
                params['year'] = year
            
            #print(f"🔍 Recherche des données agrégées pour {api_type}: {group_value}")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            #print(f"📊 {len(results)} groupes trouvés dans l'API")
            
            # Chercher notre groupe spécifique
            target_data = None
            for result in results:
                if api_type == 'country' and result.get('cntry', '').lower() == group_value.lower():
                    target_data = result.get('data')
                    break
                elif api_type == 'field' and result.get('sm-field') == group_value:
                    target_data = result.get('data')
                    break
                elif api_type == 'institution' and result.get('inst_name') == group_value:
                    target_data = result.get('data')
                    break
            
            if not target_data:
                print(f"❌ Groupe '{group_value}' non trouvé dans les résultats de l'API")
                return None, None
            
            # Décompresser si nécessaire
            if isinstance(target_data, str):
                try:
                    target_data = self.base64_decode_and_decompress(target_data)
                except Exception as e:
                    print(f"❌ Erreur de décompression: {e}")
                    return None, None
            
            # Clé pour l'année et type
            year_key = f'{prefix}_{year}'
            if year_key not in target_data:
                print(f"❌ Données non disponibles pour {year_key}")
                return None, None
            
            # Métriques de base (sans suffixe)
            metrics_base = ['nc', 'h', 'hm', 'ncs', 'ncsf', 'ncsfl', 'c']
            metrics_with_suffix = [f'{m}{suffix}' for m in metrics_base]
            
            max_limits = {}
            median_values = {}
            has_valid_data = False
            
            for i, metric_with_suffix in enumerate(metrics_with_suffix):
                base_metric = metrics_base[i]
                
                if base_metric not in target_data[year_key]:
                    print(f"Métrique {base_metric} manquante dans les données agrégées")
                    continue
                
                # STRUCTURE ATTENDUE: [min, q1, median, q3, max]
                metric_data = target_data[year_key][base_metric]
                
                if isinstance(metric_data, list) and len(metric_data) >= 5:
                    median = metric_data[2]
                    max_val = metric_data[4]
                    
                    if (isinstance(median, (int, float)) and isinstance(max_val, (int, float)) and 
                        median >= 0 and max_val >= median):
                        
                        max_limits[metric_with_suffix] = max_val
                        median_values[metric_with_suffix] = median
                        has_valid_data = True
                        #print(f"{metric_with_suffix}: median={median:.2f}, max={max_val:.2f}")
                    else:
                        print(f" Valeurs invalides pour {base_metric}: median={median}, max={max_val}")
                else:
                    print(f"⚠️ Structure invalide pour {base_metric}: {metric_data}")
            
            if has_valid_data:
                #print(f" Limites récupérées avec succès pour {group_value}")
                return max_limits, median_values
            else:
                print(f"❌ Aucune donnée valide pour {group_value}")
                return None, None
                
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des limites via API: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def interactive_author_metrics_complete(self):
        """
        Interface interactive reproduisant EXACTEMENT le dashboard serveur.
        Affiche le rank et les infos dans la figure avec les gauges.
        """
        # WIDGETS DE CONTRÔLE
        author_search = widgets.Combobox(
            value='Ioannidis, John P.A.',
            placeholder='Name...',
            options=[],
            description='Author:',
            ensure_option=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='500px')
        )
        
        search_status = widgets.HTML(value='')
        
        def update_suggestions(change):
            query = change['new']
            if len(query) >= 3:
                search_status.value = '<i>Researching...</i>'
                try:
                    results = self.search_authors(query, limit=20)
                    suggestions = [r.get('authfull', '') for r in results if r.get('authfull')]
                    author_search.options = suggestions
                    search_status.value = f'<i>✓ {len(suggestions)} results </i>'
                except Exception as e:
                    search_status.value = f'<i style="color:red">Erreur: {str(e)}</i>'
            else:
                author_search.options = []
                search_status.value = '<i>Write at least 3 caracteres...</i>'
        
        author_search.observe(update_suggestions, names='value')
        
        dataset_type = widgets.Dropdown( #RadioButtons(
            options=['Career', 'Single Year'],
            value='Career',
            description='Type:',
            style={'description_width': '100px'}
        )
        
        year_selector = widgets.Dropdown(
            options=['2017', '2018', '2019', '2020', '2021'],
            value='2021',
            description='Years:',
            style={'description_width': '100px'}
        )
        
        exclude_self = widgets.Checkbox(
            value=False,
            description='Exclude self-citation',
            style={'description_width': '100px'}
        )
        
        comparison_group = widgets.Dropdown(
            options=[
                'Max and median (red) by country', 
                'Max and median (red) by field', 
                'Max and median (red) by institute'
            ],
            value='Max and median (red) by country',
            description='Limites:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='500px')
        )
        
        update_button = widgets.Button(
            description='Generate Figures',
            button_style='success',
            icon='refresh'
        )
        
        output = widgets.Output()
        
        # FONCTION DE MISE À JOUR
        def update_plot(b):
            with output:
                clear_output(wait=True)
                
                author = author_search.value
                is_career = dataset_type.value == 'Career'
                year = year_selector.value
                exclude = exclude_self.value
                comp_group = comparison_group.value
                
                if not author:
                    print("❌ Veuillez entrer un nom d'auteur")
                    return
                
                #print(f"🔄 Searching datas for {author}...")
                
                try:
                    index = "career" if is_career else "singleyr"
                    prefix = "career" if is_career else "singleyr"
                    suffix = ' (ns)' if exclude else ''
                    
                    # Récupération des données
                    data = self.get_author_data(author, index=index)
                    if not data:
                        print(f"❌ No data found for {author}")
                        return
                    
                    key = f"{prefix}_{year}"
                    if key not in data:
                        print(f"❌ No data found for {year}")
                        return
                    
                    author_data = data[key]
                    
                    # Infos de base
                    rank = author_data.get(f'rank{suffix}', 'N/A')
                    c_score = author_data.get(f'c{suffix}', 0)
                    country = author_data.get('cntry', 'N/A')
                    field = author_data.get('sm-field', 'N/A')
                    institute = author_data.get('inst_name', 'N/A')
                    self_cit = round(author_data.get('self%', 0) * 100, 2)
                    
                    # Récupération des limites via API
                    # print(" Récupération des limites agrégées via API...")
                    max_limits, median_values = self.get_real_limits_via_api(
                        author_data, comp_group, prefix, year, suffix
                    )
                    
                    has_real_limits = max_limits is not None and median_values is not None
                    
                    
                    # CRÉATION DE LA FIGURE AVEC 7 GAUGES + RANK + INFOS
                    metric_titles = [
                        'Number of citations', 
                        'H-index', 
                        'Hm-index',
                        'Number of citations to single<br>authored papers', 
                        'Number of citations to single<br>and first authored papers', 
                        'Number of citations to single,<br>first and last authored papers',
                        'Composite score'
                    ]
                    
                    metrics_base = ['nc', 'h', 'hm', 'ncs', 'ncsf', 'ncsfl', 'c']
                    metrics_with_suffix = [f'{m}{suffix}' for m in metrics_base]
                    author_values = [author_data.get(m, 0) for m in metrics_with_suffix]
                    
                    # Créer la figure avec layout pour rank et infos
                    fig = make_subplots(
                        rows=3, cols=3,
                        specs=[
                            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'table'}],
                            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
                        ],
                        row_heights=[0.33, 0.33, 0.33],
                        vertical_spacing=0.1,
                        horizontal_spacing=0.05
                    )
                    
                    # Ajouter les 6 métriques (lignes 2 et 3)
                    for i in range(6):
                        metric = metrics_with_suffix[i]
                        title = metric_titles[i]
                        value = author_values[i]
                        
                        if i < 3:
                            row, col = 2, i + 1
                        else:
                            row, col = 3, i - 2
                        
                        # Configuration de la gauge
                        if has_real_limits and metric in max_limits and metric in median_values:
                            gauge_config = {
                                'mode': "gauge+number+delta",
                                'value': value,
                                'delta': {
                                    'reference': median_values[metric], 
                                    'increasing': {'color': "limegreen"},
                                    'decreasing': {'color': "indianred"}
                                },
                                'gauge': {
                                    'axis': {
                                        'range': [None, max_limits[metric]], 
                                        'tickcolor': "#aaa", 
                                        'tickwidth': 2
                                    },
                                    'bar': {'color': 'lightseagreen'},
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': median_values[metric]
                                    },
                                    'bgcolor': self.darkAccent1,
                                    'borderwidth': 2,
                                    'bordercolor': self.darkAccent2
                                }
                            }
                        else:
                            max_val = max(value * 2, 100) if value > 0 else 100
                            gauge_config = {
                                'mode': "gauge+number",
                                'value': value,
                                'gauge': {
                                    'axis': {
                                        'range': [None, max_val], 
                                        'tickcolor': "#aaa", 
                                        'tickwidth': 2
                                    },
                                    'bar': {'color': 'lightseagreen'},
                                    'bgcolor': self.darkAccent1,
                                    'borderwidth': 2,
                                    'bordercolor': self.darkAccent2
                                }
                            }
                        
                        fig.add_trace(
                            go.Indicator(
                                **gauge_config,
                                title={'text': title, 'font': {'color': '#aaa', 'size': 12}},
                                number={'font': {'color': 'lightseagreen', 'size': 16}}
                            ),
                            row=row, col=col
                        )
                    
                    # Ajouter le COMPOSITE SCORE (ligne 1, col 1)
                    c_metric = metrics_with_suffix[6]
                    c_value = author_values[6]
                    
                    if has_real_limits and c_metric in max_limits and c_metric in median_values:
                        c_gauge_config = {
                            'mode': "gauge+number+delta",
                            'value': c_value,
                            'delta': {
                                'reference': median_values[c_metric], 
                                'increasing': {'color': "limegreen"},
                                'decreasing': {'color': "indianred"}
                            },
                            'gauge': {
                                'axis': {
                                    'range': [None, max_limits[c_metric]], 
                                    'tickcolor': "#aaa", 
                                    'tickwidth': 2
                                },
                                'bar': {'color': 'lightseagreen'},
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': median_values[c_metric]
                                },
                                'bgcolor': self.darkAccent1,
                                'borderwidth': 2,
                                'bordercolor': self.darkAccent2
                            }
                        }
                    else:
                        max_val = max(c_value * 2, 100) if c_value > 0 else 100
                        c_gauge_config = {
                            'mode': "gauge+number",
                            'value': c_value,
                            'gauge': {
                                'axis': {
                                    'range': [None, max_val], 
                                    'tickcolor': "#aaa", 
                                    'tickwidth': 2
                                },
                                'bar': {'color': 'lightseagreen'},
                                'bgcolor': self.darkAccent1,
                                'borderwidth': 2,
                                'bordercolor': self.darkAccent2
                            }
                        }
                    
                    fig.add_trace(
                        go.Indicator(
                            **c_gauge_config,
                            title={'text': metric_titles[6], 'font': {'color': '#aaa', 'size': 14}},
                            number={'font': {'color': 'lightseagreen', 'size': 20}}
                        ),
                        row=1, col=1
                    )
                    
                    # Ajouter le RANK (ligne 1, col 2)
                    
                    fig.add_trace(
                        go.Indicator(
                            mode="number",
                            value=rank if isinstance(rank, (int, float)) else 0,
                            title={
                                'text': f"Rank of {author}", 
                                'font': {'color': 'lightseagreen', 'size': 14}
                            },
                            number={'font': {'color': 'lightseagreen', 'size': 70}}
                        ),
                        row=1, col=2
                    )
                    
                    
                    # Ajouter les INFOS comme bullet-point  (ligne 1, col 3)
                    info_text = (
                        f"• <b>Country:</b> {country}<br>"
                        f"• <b>Field:</b> {field}<br>"
                        f"• <b>Institute:</b> {institute}<br>"
                        f"• <b>Self citation:</b> {self_cit}%"
                    )

                    fig.add_annotation(
                        text=info_text,
                        xref="paper", yref="paper",
                        x=1, y=0.93,
                        showarrow=False,
                        align="left",
                        font=dict(
                            size=14,       
                            color="lightseagreen",
                            
                        )
                    )

                    #si on veut rank sous forme de tableau 
                    # fig.add_trace(
                    #     go.Table(
                
                    #         cells=dict(
                    #             values=[
                    #                 ['Country', 'Field', 'Institute', 'Self citation (%)'],
                    #                 [country, field, institute, f'{self_cit}%']
                    #             ],
                    #             fill_color=self.darkAccent1,
                    #             align='left',
                    #             font=dict(color='lightseagreen', size=11),
                    #             height=25
                    #         )
                    #     ),
                    #     row=1, col=3
                    # )
                    
                    # Mise en forme
                    status_suffix = "" if has_real_limits else " - Comparaisons non disponibles"
                    fig.update_layout(
                        height=800,
                        plot_bgcolor=self.bgc,
                        paper_bgcolor=self.bgc,
                        font=dict(color='lightseagreen', size=10),
                        title={
                            'text': f"Author Metrics - {author} ({year}){status_suffix}",
                            'font': {'size': 20, 'color': self.lightAccent1},
                            'x': 0.5,
                            'y': 0.98
                        },
                        margin={'l': 40, 'r': 40, 't': 100, 'b': 40},
                        showlegend=False
                    )
                    
                    fig.show()
                    
                    # FORMULE DU SCORE C
                    # status_note = ("⚠️ Comparaisons non disponibles" if not has_real_limits 
                    #               else f"Comparaison: {comp_group.replace('Max and median (red) by ', '')}")
                    
                    # formula_html = f"""
                    # <div style="background-color: {self.darkAccent1}; 
                    #             padding: 20px; 
                    #             border-radius: 10px;
                    #             text-align: center;
                    #             margin: 20px 0;
                    #             border: 1px solid {self.darkAccent3};
                    #             color: lightseagreen;">
                    #     <div style="font-size: 16px; margin-bottom: 10px; font-weight: bold;">
                    #         📐 Composite Score Formula
                    #     </div>
                    #     <div style="font-size: 18px; font-family: 'Courier New', monospace;">
                    #         c = (6×nc + 6×h + 5×h<sub>m</sub> + 4×nc<sub>s</sub> + 3×nc<sub>sf</sub> + 2×nc<sub>sfl</sub>) / 26
                    #     </div>
                    #     <div style="font-size: 14px; margin-top: 10px; color: {self.darkAccent3};">
                    #         Current value: <b>{c_score:.2f}</b> | {status_note}
                    #     </div>
                    # </div>
                    # """
                    # display(HTML(formula_html))
                    
                    # print("✅ Visualisation complète!")
                    
                except Exception as e:
                    print(f"❌ Erreur: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        update_button.on_click(update_plot)
        
        # LAYOUT FINAL
        controls = widgets.VBox([
            widgets.HBox([author_search, search_status, dataset_type, year_selector], layout=widgets.Layout(justify_content='flex-start', align_items='center')), #search_status]),
           # widgets.HBox([dataset_type, year_selector]),
            widgets.HBox([comparison_group, exclude_self], layout=widgets.Layout(justify_content='flex-start', align_items='center')),
            widgets.HBox([update_button], layout=widgets.Layout(justify_content='flex-start')),
        ])
         
        display(controls)
        display(output)
        
        # Chargement initial
        update_plot(None)
        
    def interactive_author_comparison(self):
        """
        Interface interactive pour comparer deux auteurs.
        Reproduit l'interface de author_vs_author_layout.
        """
        # Widgets de contrôle
        # author1_search = widgets.Text(
        #     value='Ioannidis, John P.A.',
        #     placeholder='Nom, Prénom',
        #     description='Auteur 1:',
        #     style={'description_width': '80px'},
        #     layout=widgets.Layout(width='350px')
        # )
        
        # author2_search = widgets.Text(
        #     value='Bengio, Yoshua',
        #     placeholder='Nom, Prénom',
        #     description='Auteur 2:',
        #     style={'description_width': '80px'},
        #     layout=widgets.Layout(width='350px')
        # )

        author1_search = widgets.Combobox(
            value='Ioannidis, John P.A.',
            placeholder='Commencez à taper...',
            options=[],
            description='Auteur 1:',
            ensure_option=False,
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )
        search_status1 = widgets.HTML(value='')

        def update_suggestions1(change):
            query = change['new']
            if len(query) >= 3:
                search_status1.value = '<i>🔍</i>'
                try:
                    results = self.search_authors(query, limit=20)
                    suggestions = [r.get('authfull', '') for r in results if r.get('authfull')]
                    author1_search.options = suggestions
                    search_status1.value = f'<i>✓ {len(suggestions)}</i>'
                except Exception as e:
                    search_status1.value = f'<i style="color:red">❌</i>'
            else:
                author1_search.options = []
                search_status1.value = ''

        author1_search.observe(update_suggestions1, names='value')

        author2_search = widgets.Combobox(
            value='Bengio, Yoshua',
            placeholder='Commencez à taper...',
            options=[],
            description='Auteur 2:',
            ensure_option=False,
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )
         
        search_status2 = widgets.HTML(value='')

        def update_suggestions2(change):
            query = change['new']
            if len(query) >= 3:
                search_status2.value = '<i>🔍</i>'
                try:
                    results = self.search_authors(query, limit=15)
                    suggestions = [r.get('authfull', '') for r in results if r.get('authfull')]
                    author2_search.options = suggestions
                    search_status2.value = f'<i>✓ {len(suggestions)}</i>'
                except Exception as e:
                    search_status2.value = f'<i style="color:red">❌</i>'
            else:
                author2_search.options = []
                search_status2.value = ''
        
        author2_search.observe(update_suggestions2, names='value')
        
        dataset_type = widgets.RadioButtons(
            options=['Career', 'Single Year'],
            value='Career',
            description='Type:',
            style={'description_width': '80px'}
        )
        
        year_selector = widgets.Dropdown(
            options=['2017', '2019', '2020', '2021'],
            value='2021',
            description='Année:',
            style={'description_width': '80px'}
        )
        
        exclude_self = widgets.Checkbox(
            value=False,
            description='Exclure auto-citations'
        )
        
        log_transform = widgets.Checkbox(
            value=False,
            description='Transformation log'
        )
        
        update_button = widgets.Button(
            description='Comparer',
            button_style='primary',
            icon='exchange'
        )
        
        output = widgets.Output()
        
        # Fonction de mise à jour
        def update_comparison(b):
            with output:
                clear_output(wait=True)
                
                author1 = author1_search.value
                author2 = author2_search.value
                is_career = dataset_type.value == 'Career'
                year = year_selector.value
                exclude = exclude_self.value
                log_tf = log_transform.value
                
                if not author1 or not author2:
                    print("Veuillez entrer les deux noms d'auteurs")
                    return
                
                print(f"Comparaison de {author1} et {author2}...")
                
                try:
                    fig = self.plot_author_comparison(
                        author1, author2, year, is_career, exclude, log_tf
                    )
                    fig.show()
                    
                    # Afficher les infos pour les deux auteurs
                    index = "career" if is_career else "singleyr"
                    info1 = self.get_author_info(author1, index)
                    info2 = self.get_author_info(author2, index)
                    
                    if info1 and info2:
                        print(f"\n{author1.split(',')[0]}:")
                        print(f"  Rang: {info1['rank']} | H-index: {info1['h_index']} | Citations: {info1['total_citations']}")
                        
                        print(f"\n{author2.split(',')[0]}:")
                        print(f"  Rang: {info2['rank']} | H-index: {info2['h_index']} | Citations: {info2['total_citations']}")
                
                except Exception as e:
                    print(f"❌ Erreur: {str(e)}")
        
        update_button.on_click(update_comparison)
        
        # Layout
        controls = widgets.VBox([
            widgets.HBox([author1_search, author2_search]),
            widgets.HBox([dataset_type, year_selector, update_button]),
            widgets.HBox([exclude_self, log_transform])
        ])
        
        display(controls, output)
        
        # Affichage initial
        update_comparison(None)

    # ========================================================================
    # VISUALISATIONS - CARTES ET AGRÉGATIONS
    # ========================================================================

    def get_world_map_data(self, year: str = "2021", metric: str = "h", 
                      career_type: str = "career", statistic: str = "median") -> pd.DataFrame:
        """
        Récupère les données formatées pour la carte mondiale directement depuis l'API.
        
        Args:
            year: Année (2021)
            metric: Métrique ("h", "nc", "hm", etc.)
            career_type: "career" ou "singleyr" 
            statistic: "min", "25", "median", "75", "max"
            
        Returns:
            DataFrame avec les données des pays
        """
        try:
            # Récupérer les données agrégées par pays
            aggregate_data = self.get_aggregate_data("country", 
                                                year=year if career_type == "singleyr" else None)
            
            if not aggregate_data:
                print("❌ Aucune donnée agrégée trouvée")
                return pd.DataFrame()
            
            # Mapping des statistiques
            stat_map = {'min': 0, '25': 1, 'median': 2, '75': 3, 'max': 4}
            stat_idx = stat_map.get(statistic, 2)
            
            data_list = []
            
            for country_data in aggregate_data:
                country_code = country_data.get('cntry', '').upper()
                
                # Décompresser les données si nécessaire
                if 'data' in country_data and isinstance(country_data['data'], str):
                    try:
                        decompressed_data = self.base64_decode_and_decompress(country_data['data'])
                    except Exception as e:
                        print(f"❌ Erreur décompression {country_code}: {e}")
                        continue
                else:
                    decompressed_data = country_data.get('data', {})
                
                # Clé pour l'année et type de données
                year_key = f"{career_type}_{year}"
                
                if year_key in decompressed_data:
                    year_data = decompressed_data[year_key]
                    
                    if metric in year_data:
                        metric_value = year_data[metric]
                        
                        # Extraire la valeur statistique
                        if isinstance(metric_value, list) and len(metric_value) > stat_idx:
                            value = metric_value[stat_idx]
                        else:
                            value = metric_value
                        
                        # Convertir le code pays en nom et code ISO3
                        try:
                            country_name = coco.convert(country_code, to='name_short')
                            iso_code = coco.convert(country_code, to='ISO3')
                        except:
                            country_name = country_code
                            iso_code = country_code
                        
                        data_list.append({
                            'code': iso_code,
                            'country': country_name,
                            'value': float(value),
                            'metric': metric,
                            'statistic': statistic,
                            'year': year,
                            'career_type': career_type
                        })
            
            print(f"✅ Données préparées pour {len(data_list)} pays")
            return pd.DataFrame(data_list)
            
        except Exception as e:
            print(f"❌ Erreur dans get_world_map_data: {e}")
            return pd.DataFrame()
    
    # def get_world_data(self, year: str = "2021", metric: str = "h", 
    #               career_type: str = "career", statistic: str = "median") -> pd.DataFrame:
    #     """
    #     Récupère et formate les données pour la carte mondiale
    #     """
    #     try:
    #         # Récupérer les données agrégées par pays
    #         aggregate_data = self.get_aggregate_data("country", 
    #                                             year=year if career_type == "singleyr" else None)
            
    #         if not aggregate_data:
    #             print("❌ No aggregate data found")
    #             return pd.DataFrame()
            
    #         # Mapping des statistiques
    #         stat_map = {'min': 0, '25': 1, 'median': 2, '75': 3, 'max': 4}
    #         stat_idx = stat_map.get(statistic, 2)
            
    #         # Mapping des noms de pays
    #         country_mapping = {
    #             'SUX': 'RUS',  # Russia
    #             'ANT': 'NLD',  # Netherlands
    #             'SCG': 'CZE'   # Czech Republic
    #         }
            
    #         # Préparer les données
    #         data_list = []
            
    #         for country_data in aggregate_data:
    #             country_code = country_data.get('cntry', '').upper()
                
    #             # Corriger les codes spéciaux
    #             if country_code in country_mapping:
    #                 country_code = country_mapping[country_code]
                
    #             # Obtenir le nom du pays
    #             try:
    #                 country_name = coco.convert(country_code, to='name_short')
    #                 iso_code = coco.convert(country_code, to='ISO3')
    #             except:
    #                 country_name = country_code
    #                 iso_code = country_code
                
    #             # La clé pour les données
    #             year_key = f"{career_type}_{year}"
                
    #             # Extraire la valeur de la métrique
    #             value = None
                
    #             # Chercher dans différentes structures
    #             if year_key in country_data:
    #                 data_source = country_data[year_key]
    #             elif 'data' in country_data and year_key in country_data['data']:
    #                 data_source = country_data['data'][year_key]
    #             else:
    #                 continue  # Pas de données pour cette année
                    
    #             if metric in data_source:
    #                 metric_value = data_source[metric]
    #                 value = self._extract_statistic(metric_value, stat_idx)
                
    #             if value is not None:
    #                 data_list.append({
    #                     'code': iso_code,
    #                     'country': country_name,
    #                     'value': value,
    #                     'metric': metric,
    #                     'statistic': statistic,
    #                     'year': year
    #                 })
            
    #         print(f"✅ Données préparées pour {len(data_list)} pays")
    #         return pd.DataFrame(data_list)
            
    #     except Exception as e:
    #         print(f"❌ Error in get_world_data: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return pd.DataFrame()
   
    def _extract_statistic(self, metric_value, stat_idx: int) -> float:
        """
        Extrait la valeur statistique d'une métrique.
        """
        if isinstance(metric_value, list) and len(metric_value) > stat_idx:
            return float(metric_value[stat_idx])
        elif isinstance(metric_value, (int, float)):
            return float(metric_value)
        else:
            return 0.0

    def plot_world_map(self, year: str = "2021", metric: str = "h", 
                  career_type: str = "career", statistic: str = "median") -> go.Figure:
        """
        Crée une carte choroplèthe mondiale identique au dashboard.
        
        Args:
            year: Année (2017-2021)
            metric: Métrique ("h", "nc", "hm", etc.)
            career_type: "career" ou "singleyr" 
            statistic: "min", "25", "median", "75", "max"
            
        Returns:
            Figure Plotly avec carte choroplèthe
        """
        # Récupérer les données
        df = self.get_world_map_data(year, metric, career_type, statistic)
        
        if df.empty:
            print("❌ Aucune donnée disponible pour la carte")
            return go.Figure()
        
        # Titre de la carte
        metric_name = self._get_metric_long_name(career_type == "career", metric)
        title = f"{metric_name} par pays ({year}) - {statistic}"
        
        # Créer la carte choroplèthe
        fig = px.choropleth(
            df,
            locations='code',
            color='value',
            hover_name='country',
            hover_data={
                'value': ':.2f',
                'metric': True,
                'code': False
            },
            color_continuous_scale="viridis",
            title=title,
            labels={'value': f'{metric} ({statistic})'}
        )
        
        # Appliquer le style du dashboard
        fig.update_layout(
            geo=dict(
                bgcolor=self.bgc,
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            plot_bgcolor=self.bgc,
            paper_bgcolor=self.bgc,
            font=dict(color=self.lightAccent1, size=12),
            height=600,
            margin={"r":0,"t":50,"l":0,"b":0},
            coloraxis_colorbar=dict(
                thickness=20,
                tickfont=dict(color=self.lightAccent1),
                title_font=dict(color=self.lightAccent1)
            )
        )
        
        return fig

    # def plot_world_map(self, year: str = "2021", metric: str = "h", 
    #                   career_type: str = "career", statistic: str = "median") -> go.Figure:
    #     """
    #     Crée une carte mondiale identique à celle du dashboard
        
    #     Args:
    #         year: Année (2017-2021)
    #         metric: Métrique ("h", "nc", "hm", etc.)
    #         career_type: "career" ou "singleyr" 
    #         statistic: "min", "25", "median", "75", "max"
            
    #     Returns:
    #         Figure Plotly avec carte choroplèthe
    #     """
    #     # Récupérer les données
    #     df = self.get_world_data(year, metric, career_type, statistic)
        
    #     if df.empty:
    #         print("❌ No data available for world map")
    #         return go.Figure()
        
    #     # Couleurs du dashboard
    #     darkAccent1 = '#2C2C2C'  # dark gray
    #     lightAccent1 = '#ECAB4C' # ocre
        
    #     # Créer la carte choroplèthe
    #     fig = px.choropleth(
    #         data_frame=df,
    #         locations='code',
    #         color=statistic,
    #         color_continuous_scale="viridis",
    #         hover_name='country',
    #         hover_data=['metric_name', 'metric'],
    #         title=f'{metric} by country ({year}) - {career_type}'
    #     )
        
    #     # Appliquer le style identique au dashboard
    #     fig.update_layout(
    #         autosize=True,
    #         height=700,
    #         coloraxis_colorbar_thickness=23,
    #         coloraxis_colorbar_tickfont=dict(color='white'),
    #         coloraxis_colorbar_title=dict(font={"color": 'white'}),
    #         coloraxis_colorbar_orientation="h",
    #         coloraxis_colorbar_y=-0.1,
    #         plot_bgcolor=darkAccent1,
    #         paper_bgcolor=darkAccent1,
    #         font_color='white',
    #         geo_bgcolor=darkAccent1,
    #         margin={'l': 0, 'r': 0, 'b': 0, 't': 50}
    #     )
        
    #     fig.update_geos(
    #         projection=dict(scale=1), 
    #         center=dict(lat=30), 
    #         showframe=False
    #     )
        
    #     return fig

    def plot_interactive_world_map(self, year: str = "2021", metric: str = "h", 
                                 career_type: str = "career", statistic: str = "median"):
        """
        Version interactive avec animation entre les métriques
        Similaire à la carte avec slider dans home.py
        """
        # Métriques disponibles pour l'animation
        metrics = ['h', 'nc', 'hm', 'ncs', 'ncsf', 'ncsfl', 'c']
        metrics_name = ['H-index', '# citations', 'Hm-index', 
                       '# citations to single auth papers', 
                       '# citations to single/first auth papers', 
                       '# citations to single/first/last auth papers', 
                       'Composite (c) score']
        
        # Récupérer les données pour toutes les métriques
        all_data = []
        for m in metrics:
            df = self.get_world_data(year, m, career_type, statistic)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            print("❌ No data available for interactive map")
            return go.Figure()
        
        # Combiner toutes les données
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Couleurs du dashboard
        darkAccent1 = '#2C2C2C'
        lightAccent1 = '#ECAB4C'
        
        # Créer la carte avec animation
        fig = px.choropleth(
            data_frame=combined_df,
            locations='code',
            color=statistic,
            color_continuous_scale="viridis",
            animation_frame='metric',
            hover_name='country',
            hover_data=['metric_name', 'metric'],
            title=f'World Distribution - {career_type} data up to {year}'
        )
        
        # Style identique au dashboard
        fig.update_layout(
            height=700,
            autosize=True,
            coloraxis_colorbar_thickness=23,
            coloraxis_colorbar_tickfont=dict(color='white'),
            coloraxis_colorbar_orientation="h",
            coloraxis_colorbar_y=-0.1,
            coloraxis_colorbar_title=dict(font={"color": 'white'}),
            plot_bgcolor=darkAccent1,
            paper_bgcolor=darkAccent1
        )
        
        fig.update_layout(
            sliders=[dict(
                font={'color': 'white'},
                bgcolor=lightAccent1,
                steps=[
                    {'label': 'H-index'}, 
                    {'label': '#cites'}, 
                    {'label': 'Hm-index'}, 
                    {'label': '#pprs-s'}, 
                    {'label': '#pprs-sf'},
                    {'label': '#pprs-sfl'},
                    {'label': 'C'}
                ]
            )],
            updatemenus=[dict(bgcolor=lightAccent1)]
        )
        
        fig.update_layout(
            geo_bgcolor=darkAccent1,
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
        )
        
        fig.update_geos(
            projection=dict(scale=1), 
            center=dict(lat=30), 
            showframe=False
        )
        
        return fig

    def get_country_details(self, country_code: str, year: str = "2021", 
                        career_type: str = "career") -> Dict[str, Any]:
        """
        Récupère les détails d'un pays à partir des données agrégées.
        
        Args:
            country_code: Code du pays (ex: "fr", "us")
            year: Année (2021)
            career_type: "career" ou "singleyr"
            
        Returns:
            Dictionnaire avec les données du pays
        """
        try:
            # Récupérer les données agrégées par pays
            aggregate_data = self.get_aggregate_data("country", 
                                                year=year if career_type == "singleyr" else None)
            
            if not aggregate_data:
                print("❌ Aucune donnée agrégée trouvée")
                return {}
            
            # Chercher le pays spécifique
            for country_data in aggregate_data:
                if country_data.get('cntry', '').lower() == country_code.lower():
                    print(f"✅ Pays {country_code.upper()} trouvé")
                    
                    # La clé pour les données de l'année
                    year_key = f"{career_type}_{year}"
                    
                    # Essayer différentes structures de données possibles
                    result = {
                        'country_code': country_code.upper(),
                        'year': year,
                        'career_type': career_type,
                        'year_key': year_key
                    }
                    
                    # 1. Chercher directement dans l'objet country_data
                    if year_key in country_data:
                        result['data'] = country_data[year_key]
                        result['source'] = 'direct'
                        return result
                    
                    # 2. Chercher dans le champ 'data' si présent
                    elif 'data' in country_data and year_key in country_data['data']:
                        result['data'] = country_data['data'][year_key]
                        result['source'] = 'nested_data'
                        return result
                    
                    # 3. Chercher toutes les clés qui contiennent l'année
                    else:
                        # Lister toutes les clés disponibles
                        all_keys = list(country_data.keys())
                        if 'data' in country_data and isinstance(country_data['data'], dict):
                            all_keys.extend(country_data['data'].keys())
                        
                        year_keys = [k for k in all_keys if year in k]
                        result['available_keys'] = year_keys
                        result['all_keys'] = all_keys
                        result['source'] = 'keys_listed'
                        
                        print(f"⚠️  Clé {year_key} non trouvée. Clés disponibles: {year_keys}")
                        return result
            
            print(f"❌ Pays {country_code} non trouvé dans les données agrégées")
            return {}
            
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des données du pays: {e}")
            return {}
   
    def create_world_map_interface(self):
        """
        Crée une interface interactive pour la carte mondiale
        """
            
        # Widgets de contrôle
        year_widget = widgets.Dropdown(
            options=['2017', '2018', '2019', '2020', '2021'],
            value='2021',
            description='Year:'
        )
        
        metric_widget = widgets.Dropdown(
            options=['h', 'nc', 'hm', 'ncs', 'ncsf', 'ncsfl', 'c'],
            value='h',
            description='Metric:'
        )
        
        career_widget = widgets.Dropdown(
            options=['career', 'singleyr'],
            value='career',
            description='Data Type:'
        )
        
        stat_widget = widgets.Dropdown(
            options=['min', '25', 'median', '75', 'max'],
            value='median',
            description='Statistic:'
        )
        
        update_btn = widgets.Button(description='🔄 Update Map', button_style='success')
        output = widgets.Output()
        
        def update_map(btn=None):
            with output:
                output.clear_output(wait=True)
                print("🔄 Loading world map data...")
                
                fig = self.plot_world_map(
                    year=year_widget.value,
                    metric=metric_widget.value,
                    career_type=career_widget.value,
                    statistic=stat_widget.value
                )
                
                if fig:
                    fig.show()
                else:
                    print("❌ Failed to generate map")
        
        update_btn.on_click(update_map)
        
        # Interface
        controls = widgets.HBox([year_widget, metric_widget, career_widget, stat_widget, update_btn])
        interface = widgets.VBox([
            widgets.HTML("<h2>🌍 Twopercenters World Map</h2>"),
            controls,
            output
        ])
        
        # Premier affichage
        update_map()
        
        return interface
    
    def get_author_info(self, author_name: str, index: str = "career") -> Dict[str, Any]:
        
        """
        Récupère les informations de base d'un auteur.
        
        Args:
            author_name: Nom de l'auteur
            index: "career" ou "singleyr"
            
        Returns:
            Dictionnaire avec informations (institution, pays, domaine, etc.)
        """
        results = self.search_authors(author_name, index=index, field="authfull", limit=1)
        
        if not results:
            return {}
        
        result = results[0]
        data = result.get("data", {})
        
        # Extraire la dernière année disponible
        years = [k.split('_')[1] for k in data.keys() if not k.endswith('_log')]
        if years:
            latest_year = max(years)
            latest_data = data.get(f"{index}_{latest_year}", {})
            
            return {
                "author": author_name,
                "institution": latest_data.get("inst_name", "N/A"),
                "country": latest_data.get("cntry", "N/A"),
                "field": latest_data.get("sm-field", "N/A"),
                "self_citation_pct": round(latest_data.get("self%", 0) * 100, 2),
                "rank": latest_data.get("rank", "N/A"),
                "h_index": latest_data.get("h", "N/A"),
                "total_citations": latest_data.get("nc", "N/A"),
                "latest_year": latest_year
            }
        
        return {}


#==============================================================================
#                                   Anciens tests
#==============================================================================

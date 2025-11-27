
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
import plotly.io as pio
from tqdm.auto import tqdm

pio.renderers.default = "plotly_mimetype"
class TwoPercentersClient:
    """Client to query the Twopercenters API and create visualizations"""
    
    def __init__(self, base_url: str = "https://twopercenters.db.neurolibre.org/api/v1"):
        """
        Initialize the API client.

        Args:
          base_url: Base URL of the API
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Dashboard colors
        self.darkAccent1 = '#2C2C2C'
        self.darkAccent2 = '#5b5959'
        self.darkAccent3 = '#CFCFCF'
        self.lightAccent1 = '#ECAB4C'
        self.highlight1 = 'lightsteelblue'
        self.highlight2 = 'cornflowerblue'
        self.bgc = self.darkAccent1
        
        self._author_cache = {}
        
    # ========================================================================
    # Utils methods
    # ========================================================================
    
    def _get_metric_long_name(self, career: bool, metric: str, yr: int, include_year: bool ) -> str:
        yrs = [2017, 2018, 2019, 2020, 2021]
        if yr == 0: year = 2017
        if yr != 0 and career == False: yr = yr + 1
        year = yrs[yr]
        if include_year == True: metric_name_dict = {
            'authfull':'author name',
            'inst_name':'institution name (large institutions only)',
            'cntry':'country associated with most recent institution',
            'np':f'number of papers',
            'firstyr':'year of first publication',
            'lastyr':'year of most recent publication',
            'rank (ns)':'rank based on composite score c', 
            'nc (ns)':f'total cites ', 
            'h (ns)':f'h-index', 
            'hm (ns)':f'hm-index',
            'nps (ns)':'number of single authored papers',
            'ncs (ns)':'total cites to single authored papers', 
            'cpsf (ns)':'number of single + first authored papers', 
            'ncsf (ns)':'total cites to single + first authored papers', 
            'npsfl (ns)':'number of single + first + last authored papers', 
            'ncsfl (ns)':'total cites to single + first + last authored papers',
            'c (ns)':'composite score', 
            'npciting (ns)':'number of distinct citing papers', 
            'cprat (ns)':'ratio of total citations to distinct citing papers', 
            'np cited (ns)':f'number of papers 1960-{year} that have been cited at least once (1996-{year})',
            'self%':'self-citation percentage', 
            'rank':'rank based on composite score c', 
            'nc':f'total cites', 
            'h':f'h-index',
            'hm':f'hm-index', 
            'nps':'number of single authored papers',
            'ncs':'total cites to single authored papers', 
            'cpsf':'number of single + first authored papers', 
            'ncsf':'total cites to single + first authored papers', 
            'npsfl':'number of single + first + last authored papers', 
            'ncsfl':'total cites to single + first + last authored papers',
            'c':'composite score', 
            'npciting':'number of distinct citing papers', 
            'cprat':'ratio of total citations to distinct citing papers', 
            'np cited':f'number of papers 1960-{year} that have been cited at least once (1996-{year})',
            'np_d':f'# papers 1960-{year} in titles that are discontinued in Scopus', 
            'nc_d':f'total cites 1996-{year} from titles that are discontinued in Scopus', 
            'sm-subfield-1':'top ranked Science-Metrix category (subfield) for author', 
            'sm-subfield-1-frac':'associated category fraction',
            'sm-subfield-2':'second ranked Science-Metrix category (subfield) for author', 
            'sm-subfield-2-frac':'associated category fraction', 
            'sm-field':'top ranked higher-level Science-Metrix category (field) for author', 
            'sm-field-frac':'associated category fraction',
            'rank sm-subfield-1':'rank of c within category sm-subfield-1', 
            'rank sm-subfield-1 (ns)':'rank of c (ns) within category sm-subfield-1', 
            'sm-subfield-1 count':'total number of authors within category sm-subfield-1'}
        else: metric_name_dict = {
            'authfull':'author name',
            'inst_name':'institution name (large institutions only)',
            'cntry':'country associated with most recent institution',
            'np':f'number of papers',
            'firstyr':'year of first publication',
            'lastyr':'year of most recent publication',
            'rank (ns)':'rank based on composite score c', 
            'nc (ns)':f'total cites', 
            'h (ns)':f'h-index', 
            'hm (ns)':f'hm-index',
            'nps (ns)':'number of single authored papers',
            'ncs (ns)':'total cites to single authored papers', 
            'cpsf (ns)':'number of single + first authored papers', 
            'ncsf (ns)':'total cites to single + first authored papers', 
            'npsfl (ns)':'number of single + first + last authored papers', 
            'ncsfl (ns)':'total cites to single + first + last authored papers',
            'c (ns)':'composite score', 
            'npciting (ns)':'number of distinct citing papers', 
            'cprat (ns)':'ratio of total citations to distinct citing papers', 
            'np cited (ns)':f'number of papers published since 1960 that have been cited at least', # since 1996 for career wide!
            'self%':'self-citation percentage', 
            'rank':'rank based on composite score c', 
            'nc':f'total cites', 
            'h':f'h-index',
            'hm':f'hm-index', 
            'nps':'number of single authored papers',
            'ncs':'total cites to single authored papers', 
            'cpsf':'number of single + first authored papers', 
            'ncsf':'total cites to single + first authored papers', 
            'npsfl':'number of single + first + last authored papers', 
            'ncsfl':'total cites to single + first + last authored papers',
            'c':'composite score', 
            'npciting':'number of distinct citing papers', 
            'cprat':'ratio of total citations to distinct citing papers', 
            'np cited':f'number of papers published since 1960 that have been cited at least', # since 1996 for career wide!
            'np_d':f'# papers since 1960 in titles that are discontinued in Scopus', 
            'nc_d':f'total cites since 1996 from titles that are discontinued in Scopus', 
            'sm-subfield-1':'top ranked Science-Metrix category (subfield) for author', 
            'sm-subfield-1-frac':'associated category fraction',
            'sm-subfield-2':'second ranked Science-Metrix category (subfield) for author', 
            'sm-subfield-2-frac':'associated category fraction', 
            'sm-field':'top ranked higher-level Science-Metrix category (field) for author', 
            'sm-field-frac':'associated category fraction',
            'rank sm-subfield-1':'rank of c within category sm-subfield-1', 
            'rank sm-subfield-1 (ns)':'rank of c (ns) within category sm-subfield-1', 
            'sm-subfield-1 count':'total number of authors within category sm-subfield-1'}
        return metric_name_dict.get(metric, metric)
    
    def get_es_results(self, search_term, idx_name, search_fields, exact=False, debug=False):
        """
        Search the API for aggregates (countries, fields, institutions) on the client side.
        """
        try:
            # Set the API URL according to the type of search
            url_map = {
                'career_cntry': 'aggregate/country',
                'singleyr_cntry': 'aggregate/country',
                'career_field': 'aggregate/field',
                'singleyr_field': 'aggregate/field',
                'career_inst': 'aggregate/institution',
                'singleyr_inst': 'aggregate/institution'
            }
            
            # Extract the aggregate type from idx_name
            if 'country' in idx_name or 'cntry' in idx_name:
                api_type = 'country'
            elif 'field' in idx_name:
                api_type = 'field'
            elif 'inst' in idx_name:
                api_type = 'institution'
            else:
                api_type = idx_name
                
            url = f"{self.base_url}/aggregate/{api_type}"

            # Request parameters
            params = {'limit': 500}
            
            if exact:
                params[search_fields] = search_term  # exact research 
            else:
                params['query'] = search_term       # Fuzzy / general search

            # HTTP GET request to the API
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get('results', [])

            if debug and results:
                # print(f"DEBUG - First result:")
                # print(f"  Keys: {results[0].keys()}")
                if 'data' in results[0]:
                    data_sample = str(results[0]['data'])[:100]
                    # print(f"  'data' type: {type(results[0]['data'])}")
                    # print(f"  Starting 'data': {data_sample}...")

            if results:
                return pd.json_normalize(results)
            else:
                return None

        except Exception as e:
            print(f"Erreur ES: {e}")
            return None

    def get_aggregate_data(self, group, group_name, prefix):
        """
        Retrieves aggregated data for a given group (country, field, institution)
        Returns a tuple (data, count) where count is the number of authors      
        """
        try:
            # Determine the type of aggregation for the API
            if group == 'cntry':
                api_type = 'country'
                # Convert the country name to ISO3 code as the server does
                try:
                    search_term = coco.convert(names=group_name, to='ISO3').lower()
                except:
                    search_term = group_name.lower()
            elif group == "sm-field":
                api_type = 'field'
                search_term = group_name
            elif group == "inst_name":
                api_type = 'institution'
                search_term = group_name
            else:
                return None, 0

            # Call the aggregation API
            url = f"{self.base_url}/aggregate/{api_type}"
            params = {'limit': 500}  # Increase the limit to find the specific group
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            # Locate the specific group
            group_data = None
            for result in results:
                if group == 'cntry' and result.get('cntry', '').lower() == search_term:
                    group_data = result
                    break
                elif group == 'sm-field' and result.get('sm-field') == search_term:
                    group_data = result
                    break
                elif group == 'inst_name' and result.get('inst_name') == search_term:
                    group_data = result
                    break
            
            if not group_data:
                print(f"❌ Group '{group_name}' not found in the results")
                return None, 0
            
            # Extract the data and count
            data_field = group_data.get('data')
            count = group_data.get('count', 0)
            
            # Decompress the data if necessary
            if isinstance(data_field, str) and len(data_field) > 10:
                try:
                    decompressed_data = self.base64_decode_and_decompress(data_field, flg=False)
                    return decompressed_data, count
                except Exception as e:
                    print(f"Decompression error for {group_name}: {e}")
                    return None, count
            elif isinstance(data_field, dict):
                # If it is already a dictionary, return it directly
                return data_field, count
            else:
                print(f"Unexpected data type for {group_name}: {type(data_field)}")
                return None, count
                
        except Exception as e:
            print(f"Error in get_aggregate_data for {group_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def base64_decode_and_decompress(self, encoded_data, flg=False):
        """
        Decode compressed data – client-side version
        IMPORTANT: flg=True only if the data is in a pandas Series. 
        Default is flg=False because we receive strings directly from the API.
        """
        try:
            if flg:
                if hasattr(encoded_data, '__getitem__') and not isinstance(encoded_data, str):
                    encoded_data = encoded_data[0]
            
            # Check that we have a non-empty string:
            if not isinstance(encoded_data, str):
                raise ValueError(f"encoded_data must be a string, got: {type(encoded_data)}")
            
            if len(encoded_data) < 4:
                raise ValueError(f"encoded_data too short:  {len(encoded_data)} caractères")
            
            # Base64 decode the data
            compressed_data = base64.b64decode(encoded_data)
            
            # Decompress the data using zlib
            decompressed_data = zlib.decompress(compressed_data)
            
            # Convert the decompressed string back to a dictionary
            decoded_data = json.loads(decompressed_data.decode('utf-8'))
            
            return decoded_data
            
        except Exception as e:
            print(f"Error during decompression: {e}")
            print(f" Received data type: {type(encoded_data)}")
            if isinstance(encoded_data, str):
                print(f"  Length: {len(encoded_data)}")
                print(f"  Start: {encoded_data[:50] if len(encoded_data) > 50 else encoded_data}")
            return None
    
    # ========================================================================
    # API Query Methods
    # ========================================================================
    
    def search_authors(self, query: str, index: str = "career", 
                      field: str = "authfull", limit: int = 10) -> List[Dict]:
        """Search for authors in the database."""

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
        
        if "results" in data:
            for result in data["results"]:
                if "data" in result:
                    result["data"] = self.base64_decode_and_decompress(result["data"])
        
        return data["results"]
    
    def get_author_data(self, author_name: str, index: str = "career") -> Optional[Dict]:
        """Retrieve the full data for an author."""
        cache_key = f"{author_name}_{index}"
        if cache_key in self._author_cache:
            return self._author_cache[cache_key]
        
        results = self.search_authors(author_name, index=index, field="authfull", limit=5)
        
        if not results:
            return None
        
        # Look for exact match
        for result in results:
            if result.get("authfull") == author_name:
                data = result.get("data")
                self._author_cache[cache_key] = data
                return data
        
        # If no exact match, return the first result
        data = results[0].get("data") if results else None
        self._author_cache[cache_key] = data
        return data
    
    def get_author_info(self, author_name: str, index: str = "career") -> Dict[str, Any]:
        """Retrieve an author's basic information."""
        results = self.search_authors(author_name, index=index, field="authfull", limit=1)
        
        if not results:
            return {}
        
        result = results[0]
        data = result.get("data", {})
        
        # Extract the most recent available year.
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
    # VISUALIZATIONS – INDIVIDUAL AUTHOR
    # ========================================================================
    
    # Individual author gauges – Career or Single Year
    def get_real_limits_via_api(self, author_data: Dict, comp_group: str, 
                                prefix: str, year: str, suffix: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Fetch the TRUE limits (max and median) via the /aggregate/ API.

        Args:
            author_data: Author data
            comp_group: Type of comparison ('Max and median (red) by country/field/institute')
            prefix: 'career' or 'singleyr'
            year: Year
            suffix: ' (ns)' or ''
            Returns:
            Tuple (max_limits dict, median_values dict) or (None, None)
        """
        try:
            # Determine the type of aggregation and the group value.
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
                print(f"⚠️ Group value missing for {api_type}")
                return None, None
            
            # Build the API request
            url = f"{self.base_url}/aggregate/{api_type}"
            params = {'limit': 500}
            
            if prefix == 'singleyr':
                params['year'] = year
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            # Search for our specific group
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
                print(f"❌ Group '{group_value}' not found in API results")
                return None, None
            
            # Decompress if necessary
            if isinstance(target_data, str):
                try:
                    target_data = self.base64_decode_and_decompress(target_data)
                except Exception as e:
                    print(f"❌ Decompression error: {e}")
                    return None, None
            
            # Key for year and type
            year_key = f'{prefix}_{year}'
            if year_key not in target_data:
                print(f"❌ Data not available for {year_key}")
                return None, None
            
            # Base metrics (without suffix)
            metrics_base = ['nc', 'h', 'hm', 'ncs', 'ncsf', 'ncsfl', 'c']
            metrics_with_suffix = [f'{m}{suffix}' for m in metrics_base]
            
            max_limits = {}
            median_values = {}
            has_valid_data = False
            
            for i, metric_with_suffix in enumerate(metrics_with_suffix):
                base_metric = metrics_base[i]
                
                if base_metric not in target_data[year_key]:
                    print(f"Metric {base_metric} missing in aggregated data")
                    continue
                
                # EXPECTED STRUCTURE: [min, q1, median, q3, max]
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
                        print(f" Invalid values for {base_metric}: median={median}, max={max_val}")
                else:
                    print(f"⚠️ Invalid structure for {base_metric}: {metric_data}")
            
            if has_valid_data:
                return max_limits, median_values
            else:
                print(f"❌ Invalid values for {group_value}")
                return None, None
                
        except Exception as e:
            print(f"❌ Error retrieving limits via API: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def interactive_author_metrics_complete(self):
        """
        Interactive interface reproducing the server dashboard EXACTLY.
        Displays the rank and information in the figure along with the gauges.
        """
        # CONTROL WIDGETS
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
        
        dataset_type = widgets.Dropdown( 
            options=['Career', 'Single Year'],
            value='Career',
            description='Type:',
            style={'description_width': '100px'}
        )
        
        year_selector = widgets.Dropdown(
            options=['2017', '2018', '2019', '2020', '2021'],
            value='2017',
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
        
        # UPDATE FUNCTION
        def update_plot(b):
            with output:
                clear_output(wait=True)
                
                author = author_search.value
                is_career = dataset_type.value == 'Career'
                year = year_selector.value
                exclude = exclude_self.value
                comp_group = comparison_group.value
                
                if not author:
                    print("❌ Please enter an author name")
                    return
                
                
                try:
                    index = "career" if is_career else "singleyr"
                    prefix = "career" if is_career else "singleyr"
                    suffix = ' (ns)' if exclude else ''
                    
                    # Data retrieval
                    data = self.get_author_data(author, index=index)
                    if not data:
                        print(f"❌ No data found for {author}")
                        return
                    
                    key = f"{prefix}_{year}"
                    if key not in data:
                        print(f"❌ No data found for {year}")
                        return
                    
                    author_data = data[key]
                    
                    # Infos
                    rank = author_data.get(f'rank{suffix}', 'N/A')
                    c_score = author_data.get(f'c{suffix}', 0)
                    country = author_data.get('cntry', 'N/A')
                    field = author_data.get('sm-field', 'N/A')
                    institute = author_data.get('inst_name', 'N/A')
                    self_cit = round(author_data.get('self%', 0) * 100, 2)
                    
                    # Retrieval of limits via API
                    max_limits, median_values = self.get_real_limits_via_api(
                        author_data, comp_group, prefix, year, suffix
                    )
                    
                    has_real_limits = max_limits is not None and median_values is not None
                    
                    
                    # Creation of the figure with 7 gauges + rank + info
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
                    
                    # Add 6 metrics (lines 2 and 3)
                    for i in range(6):
                        metric = metrics_with_suffix[i]
                        title = metric_titles[i]
                        value = author_values[i]
                        
                        if i < 3:
                            row, col = 2, i + 1
                        else:
                            row, col = 3, i - 2
                        
                        # Gauges configuration
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
                    
                    # Add COMPOSITE SCORE (line 1, col 1)
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
                    
                    # Add RANK (line 1, col 2)
                    
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
                    
                    #Add truncation and line breaks to avoid overlapping with (1,2)
                    def wrap_text(text, max_length=30):
                        if len(text) <= max_length:
                            return text
                        
                        words = text.split()
                        lines = []
                        current_line = ""
                        
                        for word in words:
                            if len(current_line) + len(word) + 1 <= max_length:
                                current_line += (" " if current_line else "") + word
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = word
                        
                        if current_line:
                            lines.append(current_line)
                        
                        return "<br>".join(lines)
                    
                    country = wrap_text(country, 30)
                    field = wrap_text(field, 30)
                    institute = wrap_text(institute, 30)

                    # Add bullet-point infos (line 1, col 3)
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
                    
                    # Formatting
                    status_suffix = "" if has_real_limits else " - Comparaison not available"
                    fig.update_layout(
                        height=800,
                        plot_bgcolor=self.bgc,
                        paper_bgcolor=self.bgc,
                        font=dict(color='lightseagreen', size=10),
                        # title={
                        #     'text': f"{author} ({year}){status_suffix}",
                        #     'font': {'size': 20, 'color': self.lightAccent1},
                        #     'x': 0.5,
                        #     'y': 0.98
                        # },
                        margin={'l': 40, 'r': 40, 't': 100, 'b': 40},
                        showlegend=False
                    )
                    
                    fig.show()
                    
                    # composite score formula
                    # status_note = ("⚠️ Comparaisons not available" if not has_real_limits 
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
                    #          Composite Score Formula
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
                    
                except Exception as e:
                    print(f"❌ Erreur: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        update_button.on_click(update_plot)
        
        # LAYOUT FINAL
        controls = widgets.VBox([
            widgets.HBox([author_search, search_status, dataset_type, year_selector], layout=widgets.Layout(justify_content='flex-start', align_items='center')), 
            widgets.HBox([comparison_group, exclude_self], layout=widgets.Layout(justify_content='flex-start', align_items='center')),
            widgets.HBox([update_button], layout=widgets.Layout(justify_content='flex-start')),
        ])
         
        display(controls)
        display(output)
        
        # Initial loading
        update_plot(None)
        
    #metrics for one author by career and year
    def author_vs_career_plot(self, author_name: str, exclude_self_citations: bool = False, 
                          width: int = 900, height: int = 1500):
        """Create the Career vs Single-Year Comparison Chart."""
        
        # Retrieve the data using self.get_author_data().
        data_career = self.get_author_data(author_name, index='career')
        data_singleyear = self.get_author_data(author_name, index='singleyr')
        
        if not data_career and not data_singleyear:
            print(f"❌ No data for {author_name}")
            return None
        
        # Extract the years - CAREER
        years_career = []
        metrics_data_career = []
        if data_career:
            for key in sorted(data_career.keys()):
                if not key.endswith('_log'):
                    parts = key.split('_')
                    if len(parts) == 2 and parts[0] == 'career':
                        years_career.append(parts[1])
                        metrics_data_career.append(data_career[key])
        
        # Extract the years - SINGLE YEAR
        years_singleyear = []
        metrics_data_singleyear = []
        if data_singleyear:
            for key in sorted(data_singleyear.keys()):
                if not key.endswith('_log'):
                    parts = key.split('_')
                    if len(parts) == 2 and parts[0] == 'singleyr':
                        years_singleyear.append(parts[1])
                        metrics_data_singleyear.append(data_singleyear[key])
        
        # Create DataFrames
        df_career = pd.DataFrame(metrics_data_career) if metrics_data_career else pd.DataFrame()
        df_singleyear = pd.DataFrame(metrics_data_singleyear) if metrics_data_singleyear else pd.DataFrame()
        
        if not df_career.empty:
            df_career['Year'] = years_career
            df_career['self%'] = df_career['self%'] * 100
        
        if not df_singleyear.empty:
            df_singleyear['Year'] = years_singleyear
            df_singleyear['self%'] = df_singleyear['self%'] * 100
        
        # Define the metrics
        suffix = ' (ns)' if exclude_self_citations else ''
        metrics_list = [
            f'rank{suffix}', f'c{suffix}', f'nc{suffix}', f'h{suffix}', 
            f'hm{suffix}', f'ncs{suffix}', f'ncsf{suffix}', f'ncsfl{suffix}',
            'np', 'self%'
        ]
        
        # Filter available metrics 
        available_metrics = [m for m in metrics_list 
                            if (not df_career.empty and m in df_career.columns) or 
                            (not df_singleyear.empty and m in df_singleyear.columns)]
        
        if not available_metrics:
            print("❌ no metrics available")
            return None
        
        # Create the subtitles using your _get_metric_long_name function
        subplot_titles = []
        for metric in available_metrics:
            # Career (column 1) - career=True, include_year=True
            career_title = self._get_metric_long_name(career=True, metric=metric, yr=0, include_year=True)
            # Single Year (column 2) - career=False, include_year=False  
            singleyear_title = self._get_metric_long_name(career=False, metric=metric, yr=0, include_year=False)
            subplot_titles.extend([career_title, singleyear_title])
        
        # Create the figure with subplots
        fig = make_subplots(
            rows=len(available_metrics), 
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.035,
            horizontal_spacing=0.09,
            #column_titles=[f"<b>Career: {author_name}</b>", f"<b>Single Year: {author_name}</b>"]
        )
        
        # Colors
        col_turbo = px.colors.sample_colorscale("turbo", [n/99 for n in range(100)])
        col_viridis = px.colors.sample_colorscale("viridis", [n/99 for n in range(100)])
        
        # Add traces
        for i, metric in enumerate(available_metrics):
            row = i + 1
            
            # Career (column 1)
            if not df_career.empty and metric in df_career.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_career['Year'], y=df_career[metric],
                        mode='lines+markers',
                        marker=dict(color=col_turbo[99-i-25], size=8),
                        line=dict(color=col_turbo[99-i-25], width=2),
                        name=f"Career_{metric}",
                        hovertemplate='Year: %{x}<br>Value: %{y}<extra></extra>',
                        showlegend=False
                    ), 
                    row=row, col=1
                )
            
            # Single Year (column 2)
            if not df_singleyear.empty and metric in df_singleyear.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_singleyear['Year'], y=df_singleyear[metric],
                        mode='lines+markers',
                        marker=dict(color=col_turbo[99-i-25], size=8),
                        line=dict(color=col_turbo[99-i-25], width=2),
                        name=f"SingleYear_{metric}",
                        hovertemplate='Year: %{x}<br>Value: %{y}<extra></extra>',
                        showlegend=False
                    ), 
                    row=row, col=2
                )
        
        # Style
        fig.update_xaxes(
            gridcolor=self.darkAccent2, 
            linecolor=self.darkAccent2, 
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor=self.darkAccent2, 
            linecolor=self.darkAccent2, 
            zeroline=False
        )
        annotations = list(fig.layout.annotations)  # <-- keep existing subtitiles

        # Add annotations
        annotations += [
            dict(text=f"<b>Career-long data: {author_name}</b>", x=0, y=1.05,
                xref="paper", yref="paper", showarrow=False, font=dict(size=16)),
            dict(text=f"<b>Single-year data: {author_name}</b>", x=1, y=1.05,
                xref="paper", yref="paper", showarrow=False, font=dict(size=16))
        ]

        fig.update_layout(
            plot_bgcolor=self.bgc, 
            paper_bgcolor=self.bgc, 
            font=dict(color=self.lightAccent1),
            height=height, 
            width=width, 
            showlegend=False,
            annotations=annotations
        )
        
        return fig
    
    def interactive_author_vs_career_plot(self, author_name: str = 'Ioannidis, John P.A.'):
        """Create the interactive interface with widgets for the Career vs Single-Year Comparison Chart."""
        
        author_search = widgets.Combobox(
            value=author_name, 
            placeholder='Name...', 
            options=[],
            description='Author:', 
            ensure_option=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='500px')
        )
        
        search_status = widgets.HTML(value='')
        exclude_self = widgets.Checkbox(value=False, description='Exclude self-citation')
        update_button = widgets.Button(description='Generate Plot', button_style='success')
        output = widgets.Output()
        
        def update_suggestions(change):
            query = change['new']
            if len(query) >= 3:
                # Use self.search_authors()
                results = self.search_authors(query, limit=20)
                author_search.options = [r.get('authfull', '') for r in results]
                search_status.value = f'✓ {len(results)} résultats'
            else:
                author_search.options = []
        
        author_search.observe(update_suggestions, names='value')
        
        def update_plot(b=None):
            with output:
                clear_output(wait=True)
                fig = self.author_vs_career_plot(author_search.value, exclude_self.value)
                if fig:
                    fig.show()
        
        update_button.on_click(update_plot)
        
        controls = widgets.VBox([
            widgets.HBox([author_search, search_status, exclude_self]),
            update_button
        ])
        
        display(controls, output)
        update_plot()

    # ========================================================================
    # COMPARISONS 
    # ========================================================================
   
    #author_vs_author_layout
    def plot_author_comparison(self, author1: str, author2: str, 
                              year: str = "2021", career: bool = True,
                              exclude_self_citations: bool = False,
                              log_transform: bool = False) -> go.Figure:
        """
        Compare two authors on their metrics using bar plot.
        Reproduces the visualization from author_vs_author_layout.
        """
        index = "career" if career else "singleyr"
        prefix = "career" if career else "singleyr"
        
        # Retrieve the data
        data1 = self.get_author_data(author1, index=index)
        data2 = self.get_author_data(author2, index=index)
        
        if not data1 or not data2:
            print(" Unable to retrieve data for one of the authors")
            return go.Figure()
        
        # Extract the data for the year 
        key = f"{prefix}_{year}"
        if key not in data1 or key not in data2:
            print(f" Unavailable data for the year {year}")
            return go.Figure()
        
        metrics1 = data1[key]
        metrics2 = data2[key]
        
        # Define the metrics to compare
        suffix = ' (ns)' if exclude_self_citations else ''
        metrics_list = [f'nc{suffix}', f'h{suffix}', f'hm{suffix}', 
                       f'ncs{suffix}', f'ncsf{suffix}', f'ncsfl{suffix}']
        
        metric_titles = ['Citations', 'H-index', 'Hm-index',
                        'Cit. Single', 'Cit. Single+First', 'Cit. Single+First+Last']
        
        # Create the figure with subplot 
        fig = make_subplots(
            rows=1, cols=6,
            subplot_titles=metric_titles,
            horizontal_spacing=0.02
        )
        
        # Add bar plot for each metrics 
        for i, (metric, title) in enumerate(zip(metrics_list, metric_titles)):
            y1 = metrics1.get(metric, 0)
            y2 = metrics2.get(metric, 0)
            
            # Apply log transformation if requested
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
        
        # Formating
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
    
    def interactive_author_comparison(self):
        """Interactive interface to compare two authors with bar plot."""
        
        # ==========================================================================================
        # CONTROLE WIDGETS
        # ==========================================================================================
        
        # 1st Author selected
        author1_search = widgets.Combobox(
            value='Ioannidis, John P.A.',
            placeholder='Start typing name and surname...',
            options=[],
            description='Author 1:',
            ensure_option=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='500px')
        )
        
        search_status1 = widgets.HTML(value='')
        
        # Data type for author 1
        career_single_a1 = widgets.Dropdown(
            options=['Career', 'Single Year'],
            value='Career',
            description='Type:',
            style={'description_width': '100px'}
        )
        
        # Year for author 1
        year_a1 = widgets.Dropdown(
            options=['2017', '2018', '2019', '2020', '2021'],
            value='2017',
            description='Year:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='150px')
        )
        
        # 2nd Author selected 
        author2_search = widgets.Combobox(
            value='Bengio, Yoshua',
            placeholder='Start typing name and surname...',
            options=[],
            description='Author 2:',
            ensure_option=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='500px')
        )
        
        search_status2 = widgets.HTML(value='')
        
        # Data type for author 2
        career_single_a2 = widgets.Dropdown( 
            options=['Career', 'Single Year'],
            value='Career',
            description='Type:',
            style={'description_width': '100px'}
        )
        
        # Year for author 2
        year_a2 = widgets.Dropdown(
            options=['2017', '2018', '2019', '2020', '2021'],
            value='2017',
            description='Year:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='150px')
        )
        
        # General option
        exclude_self = widgets.Checkbox(
            value=False,
            description='Exclude self-citations',
            style={'description_width': '100px'}
        )
        
        log_transform = widgets.Checkbox(
            value=False,
            description='Log transformed',
            style={'description_width': '120px'}
        )
        
        update_button = widgets.Button(
            description='Compare Authors',
            button_style='success',
            icon='refresh',
            layout=widgets.Layout(width='200px')
        )
        
        output = widgets.Output()
        
        # ==========================================================================================
        # UPDATE FUNCTION
        # ==========================================================================================
        
        def update_suggestions1(change):
            query = change['new']
            if len(query) >= 3:
                search_status1.value = '<i>Researching...</i>'
                try:
                    results = self.search_authors(query, limit=20)
                    suggestions = [r.get('authfull', '') for r in results if r.get('authfull')]
                    author1_search.options = suggestions
                    search_status1.value = f'<i>✓ {len(suggestions)} results found</i>'
                except Exception as e:
                    search_status1.value = f'<i style="color:red">Error: {str(e)}</i>'
            else:
                author1_search.options = []
                search_status1.value = '<i>Write at least 3 characters...</i>'
        
        def update_suggestions2(change):
            query = change['new']
            if len(query) >= 3:
                search_status2.value = '<i>Researching...</i>'
                try:
                    results = self.search_authors(query, limit=20)
                    suggestions = [r.get('authfull', '') for r in results if r.get('authfull')]
                    author2_search.options = suggestions
                    search_status2.value = f'<i>✓ {len(suggestions)} results found</i>'
                except Exception as e:
                    search_status2.value = f'<i style="color:red">Error: {str(e)}</i>'
            else:
                author2_search.options = []
                search_status2.value = '<i>Write at least 3 characters...</i>'
        
        author1_search.observe(update_suggestions1, names='value')
        author2_search.observe(update_suggestions2, names='value')
        
        # ==========================================================================================
        # COMPARAISON MAIN FUNCTION
        # ==========================================================================================

        def create_comparison_figures(author1, career1, year1, author2, career2, year2, exclude_self, log_transform):
            """Create one figure with 7 plots + RANKs."""
            
            prefix1 = 'career' if career1 else 'singleyr'
            prefix2 = 'career' if career2 else 'singleyr'
            
            data1 = self.get_author_data(author1, prefix1)
            data2 = self.get_author_data(author2, prefix2)
            
            if not data1 or not data2:
                return None
            
            key1 = f"{prefix1}_{year1}"
            key2 = f"{prefix2}_{year2}"
            
            if key1 not in data1 or key2 not in data2:
                return None
            
            metrics1 = data1[key1]
            metrics2 = data2[key2]
            
            suffix = ' (ns)' if exclude_self else ''
            metrics_list = ['nc', 'h', 'hm', 'ncs', 'ncsf', 'ncsfl', 'c']
            metrics_with_suffix = [f'{m}{suffix}' for m in metrics_list]
            
            values1 = [metrics1.get(m, 0) for m in metrics_with_suffix]
            values2 = [metrics2.get(m, 0) for m in metrics_with_suffix]
            
            rank1 = metrics1.get(f'rank{suffix}', 'N/A')
            rank2 = metrics2.get(f'rank{suffix}', 'N/A')
            
            fig = make_subplots(
                rows=2, cols=6,
                specs=[
                    # Line 1: Score C (1,1) + Rank A1 (1,3) + Rank A2 (1,5) 
                    [{'type': 'bar'}, None, {'type': 'indicator'}, None, {'type': 'indicator'}, None],
                    # Line 2:  6 plot, 1 per metric side by side 
                    [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]
                ],
                row_heights=[0.6, 0.6],
                column_widths=[0.16, 0.16, 0.16, 0.16, 0.16, 0.16],
                vertical_spacing=0.15,
                horizontal_spacing=0.02
            )
            
            # ==================================================================
            # LINE 1: Score C (1,1) + Rank A1 (1,3) + Rank A2 (1,5)
            # ==================================================================
            
            # Score C (1,1)
            c_value1 = values1[6]  # Score C author 1
            c_value2 = values2[6]  # Score C author 2
            
            # Log if selected
            y1_c = c_value1
            y2_c = c_value2
            if log_transform:
                y1_c = np.log1p(y1_c) if y1_c > 0 else 0
                y2_c = np.log1p(y2_c) if y2_c > 0 else 0
            
            # Author1 - Score C
            fig.add_trace(
                go.Bar(
                    x=[''],  # No label on the X-axis.
                    y=[y1_c],
                    marker_color=self.highlight1,
                    marker_line_width=0,
                    showlegend=True,
                    legendgroup='author1',
                    name=author1, 
                    text=[f"{c_value1:.1f}"],
                    textposition='auto',
                    hovertemplate=f'Composite Score (C)<br>{author1}: %{{text}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Author 2 - Score C
            fig.add_trace(
                go.Bar(
                    x=[''],  # No label on the X-axis.
                    y=[y2_c],
                    marker_color=self.highlight2,
                    marker_line_width=0,
                    showlegend=True,
                    legendgroup='author2',
                    name=author2, 
                    text=[f"{c_value2:.1f}"],
                    textposition='auto',
                    hovertemplate=f'Composite Score (C)<br>{author2}: %{{text}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Author 1 rank  (1,3)
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=rank1 if isinstance(rank1, (int, float)) else 0,
                    number={'font': {'color': self.highlight1, 'size': 50}},
                    title={
                        'text': f"<b>Rank<br>{author1}</b>",
                        'font': {'color': self.highlight1, 'size': 14}
                    }
                ),
                row=1, col=3
            )
            
            # Author 2 rank  (1,5)
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=rank2 if isinstance(rank2, (int, float)) else 0,
                    number={'font': {'color': self.highlight2, 'size': 50}},
                    title={
                        'text': f"<b>Rank<br>{author2}</b>",
                        'font': {'color': self.highlight2, 'size': 14}
                    }
                ),
                row=1, col=5
            )
            
            # ==================================================================
            # LINE 2: 6 metric side by side  (2,1 à 2,6)
            # ==================================================================
            
            metric_titles = ['NC', 'H', 'Hm', 'NCS', 'NCSF', 'NCSFL']
            full_titles = [
                'Number of citations',
                'H-index', 
                'Hm-index',
                'Number of citations to single authored papers',
                'Number of citations to single and first authored papers',
                'Number of citations to single, first and last authored papers'
            ]
            
            for i in range(6):
                y1 = values1[i]
                y2 = values2[i]
                
                if log_transform:
                    y1 = np.log1p(y1) if y1 > 0 else 0
                    y2 = np.log1p(y2) if y2 > 0 else 0
                
                # Author 1
                fig.add_trace(
                    go.Bar(
                        x=[''],  # No label on the X-axis.
                        y=[y1],
                        marker_color=self.highlight1,
                        marker_line_width=0,
                        showlegend=False,  
                        legendgroup='author1',  # Link all traces of Author 1
                        name=author1, 
                        text=[f"{values1[i]:.0f}"],
                        textposition='auto',
                        hovertemplate=f'{full_titles[i]}<br>{author1}: %{{text}}<extra></extra>' 
                    ),
                    row=2, col=i+1
                )
                
                # Author 2
                fig.add_trace(
                    go.Bar(
                        x=[''],  #  No label on the X-axis.
                        y=[y2],
                        marker_color=self.highlight2,
                        marker_line_width=0,
                        showlegend=False,  
                        legendgroup='author2',  # Link all traces of Author 2
                        name=author2, 
                        text=[f"{values2[i]:.0f}"],
                        textposition='auto',
                        hovertemplate=f'{full_titles[i]}<br>{author2}: %{{text}}<extra></extra>'
                    ),
                    row=2, col=i+1
                )
            
            # ==================================================================
            # Final layout
            # ==================================================================
            
            fig.update_layout(
                height=450,
                plot_bgcolor=self.bgc,
                paper_bgcolor=self.bgc,
                font={'color': self.lightAccent1},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.15,
                    xanchor="center",
                    x=0.5,
                    font={'size': 14},
                    itemclick='toggle',  
                    itemdoubleclick='toggleothers'  
                ),
                margin={'l': 20, 'r': 20, 't': 120, 'b': 50},
                bargap=0.3,  
                bargroupgap=0.1
            )
            
            for row in [1, 2]:
                for col in [1, 2, 3, 4, 5, 6]:
                    if row == 1 and col != 1:  # Skip empty columns in row 1.
                        continue
                        
                    fig.update_xaxes(
                        showgrid=False,
                        showticklabels=False,  
                        tickangle=0,
                        row=row, col=col
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridcolor=self.darkAccent2,
                        linecolor=self.darkAccent2,
                        row=row, col=col
                    )
            
            # Subtitles 
            annotations = []
            
            # Score C title (line 1, col 1)
            annotations.append(dict(
                x=0.07, y=1.08,
                xref='paper', yref='paper',
                text='<b>Composite Score (C)</b>',
                showarrow=False,
                font=dict(size=12, color=self.lightAccent1),
                xanchor='center'
            ))
            
            #  metrics titles (line 2)
            metric_positions = (0.05), (0.21), (0.39), (0.56), (0.75), (0.92)
            for i, title in enumerate(metric_titles):
                annotations.append(dict(
                    x=metric_positions[i], y=0.50,
                    xref='paper', yref='paper',
                    text=f'<b>{title}</b>',
                    showarrow=False,
                    font=dict(size=12, color=self.lightAccent1),
                    xanchor='center'
                ))
            
            fig.update_layout(annotations=annotations)
            
            return fig

        def update_comparison(b=None):
            with output:
                clear_output(wait=True)
                
                author1 = author1_search.value
                author2 = author2_search.value
                career1 = career_single_a1.value == 'Career'
                career2 = career_single_a2.value == 'Career'
                year1 = year_a1.value
                year2 = year_a2.value
                exclude = exclude_self.value
                log_tf = log_transform.value
                
                if not author1 or not author2:
                    print("❌ Please enter both author names")
                    return
                
                try:
                    fig = create_comparison_figures(
                        author1, career1, year1, author2, career2, year2, exclude, log_tf
                    )
                    
                    if fig:
                        fig.show()
                    else:
                        print("❌ No data available for one or both authors")
                        
                except Exception as e:
                    print(f"❌ Error generating comparison: {str(e)}")
                    import traceback
                    traceback.print_exc()

        update_button.on_click(update_comparison)
        
        # ==========================================================================================
        # LAYOUT FINAL
        # ==========================================================================================
        
        # 1st author
        author1_controls = widgets.VBox([
            widgets.HBox([author1_search, search_status1]),
            widgets.HBox([career_single_a1, year_a1])
        ])
        
        # 2nd author
        author2_controls = widgets.VBox([
            widgets.HBox([author2_search, search_status2]),
            widgets.HBox([career_single_a2, year_a2])
        ])
        
        # Global options
        global_controls = widgets.HBox([
            exclude_self,
            log_transform,
            update_button
        ])
        
        # MAin Layout 
        main_controls = widgets.VBox([
            widgets.HBox([author1_controls, author2_controls]),
            global_controls
        ])
        
        display(main_controls)
        display(output)
        update_comparison()

    #author_vs_group_layout
    def interactive_author_vs_group_comparison(self):
        """Interactive interface for comparing an author to a group with 7 box plot + RANK + number of authors of the chosen group."""
        
        # ==========================================================================================
        # CONTROL WIDGET - AUTHOR
        # ==========================================================================================
        
        author_search = widgets.Combobox(
            value='Ioannidis, John P.A.',
            placeholder='Start typing name and surname...',
            options=[],
            description='Author:',
            ensure_option=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='500px')
        )
        
        search_status = widgets.HTML(value='')
        
        career_single_author = widgets.Dropdown(
            options=['Career', 'Single Year'],
            value='Career',
            description='Dataset:',
            style={'description_width': '100px'}
        )
        
        year_author = widgets.Dropdown(
            options=['2017', '2018', '2019', '2020', '2021'],
            value='2017',
            description='Year:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='150px')
        )
        
        # ==========================================================================================
        # CONTROL WIDGET  - GROUP
        # ==========================================================================================
        
        group_type = widgets.Dropdown(
            options=[
                ('Country', 'cntry'),
                ('Field', 'sm-field'), 
                ('Institution', 'inst_name')
            ],
            value='sm-field',
            description='Group:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        group_selection = widgets.Combobox(
            value='Clinical Medicine',
            placeholder='Select group...',
            options=[],
            description='Select:',
            ensure_option=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        group_status = widgets.HTML(value='')
        
        # ==========================================================================================
        # GENERAL OPTIONS
        # ==========================================================================================
        
        exclude_self = widgets.Checkbox(
            value=False,
            description='Exclude self-citations',
            style={'description_width': '100px'}
        )
        
        log_transform = widgets.Checkbox(
            value=False,
            description='Log transformed',
            style={'description_width': '150px'}
        )
        
        update_button = widgets.Button(
            description='Generate Comparison',
            button_style='success',
            icon='refresh',
            layout=widgets.Layout(width='250px')
        )
        
        output = widgets.Output()
        
        # ==========================================================================================
        # UPDATE FUNCTIONS
        # ==========================================================================================
        
        def update_author_suggestions(change):
            query = change['new']
            if len(query) >= 3:
                search_status.value = '<i>Researching...</i>'
                try:
                    results = self.search_authors(query, limit=20)
                    suggestions = [r.get('authfull', '') for r in results if r.get('authfull')]
                    author_search.options = suggestions
                    search_status.value = f'<i>✓ {len(suggestions)} results found</i>'
                except Exception as e:
                    search_status.value = f'<i style="color:red">Error: {str(e)}</i>'
            else:
                author_search.options = []
                search_status.value = '<i>Write at least 3 characters...</i>'
        
        def update_group_options(change):
            try:
                career = career_single_author.value == 'Career'
                year = year_author.value
                group = group_type.value
                
                if group == 'cntry':
                    api_type = 'country'
                elif group == 'sm-field':
                    api_type = 'field'
                else:  # inst_name
                    api_type = 'institution'
                
                url = f"{self.base_url}/aggregate/{api_type}"
                params = {'limit': 500}
                
                if not career:
                    params['year'] = year
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                results = data.get('results', [])
                
                options = []
                for result in results:
                    if group == 'cntry':
                        code = result.get('cntry', '').lower()
                        if code and code not in ['csk', 'nan']:
                            try:
                                name = coco.convert(code, to='name_short')
                                if name:
                                    options.append(name)
                            except:
                                pass
                    elif group == 'sm-field':
                        field = result.get('sm-field', '')
                        if field and field != 'Nan':
                            options.append(field)
                    else:  # inst_name
                        inst = result.get('inst_name', '')
                        if inst and inst != 'Nan':
                            options.append(inst)
                
                group_selection.options = sorted(set(options))
                group_status.value = f'<i>✓ {len(options)} groups available</i>'
                
            except Exception as e:
                group_status.value = f'<i style="color:red">Error: {str(e)}</i>'
        
        author_search.observe(update_author_suggestions, names='value')
        group_type.observe(update_group_options, names='value')
        career_single_author.observe(update_group_options, names='value')
        year_author.observe(update_group_options, names='value')
        
        # ==========================================================================================
        # MAIN FUNCTION 
        # ==========================================================================================
        
        def create_comparison_figures_author_group(author_name, career_author, year_val, group_type_val, 
                                    group_name, exclude_self_val, log_transform_val):
            """Create comparison figure with box plot per group and author value"""
            
            try:
                # Retrieve the author’s data.
                prefix = 'career' if career_author else 'singleyr'
                author_data = self.get_author_data(author_name, prefix)
                
                if not author_data:
                    print(f"❌ no data found for {author_name}")
                    return None
                    
            
                key = f"{prefix}_{year_val}"
                if key not in author_data:
                    print(f"❌ year {year_val} not available for this author")
                    return None
                    
                author_metrics = author_data[key]
                

                group_data, count = self.get_aggregate_data(group_type_val, group_name, prefix)

                if group_data is None or key not in group_data:
                    print(f"❌ Data not available for the year {year_val}")
                    return None

                group_stats = group_data[key]

               
                try:
                    n2 = group_stats['c'][5]  # number of author is the 5th index 
                except:
                    n2 = 0

                suffix = ' (ns)' if exclude_self_val else ''
                metrics_base = ['nc', 'h', 'hm', 'ncs', 'ncsf', 'ncsfl', 'c']
                metrics_with_suffix = [f'{m}{suffix}' for m in metrics_base]
                
                metric_titles = [
                    'NC', 
                    'H', 
                    'Hm', 
                    'NCS', 
                    'NCSF', 
                    'NCSFL', 
                    'Composite score (C)'
                ]
                
              
                fig = make_subplots(
                    rows=2,
                    cols=6,
                    specs=[
                        [
                            {'type': 'box'}, None,
                            {'type': 'indicator'}, None,
                            {'type': 'indicator'}, None
                        ],
                        [
                            {'type': 'box'}, {'type': 'box'}, {'type': 'box'},
                            {'type': 'box'}, {'type': 'box'}, {'type': 'box'}
                        ]
                    ],
                    row_heights=[0.45, 0.45],
                    horizontal_spacing=0.05,
                    vertical_spacing=0.15,
                    column_widths=[0.16]*6,

                )

                
                # ==================================================================
                # LINE 1: Score C (1,1) + Rank (1,2) + Number of authors (1,3)
                # ==================================================================
                
                # Score C - Box plot 
                c_metric = metrics_base[6]
                c_value_author = author_metrics.get(metrics_with_suffix[6], 0)
                
                if c_metric in group_stats:
                    c_stats = group_stats[c_metric]
                    if isinstance(c_stats, list) and len(c_stats) >= 5:
                        fig.add_trace(
                            go.Box(
                                q1=[c_stats[1]],
                                median=[c_stats[2]],
                                q3=[c_stats[3]],
                                lowerfence=[c_stats[0]],
                                upperfence=[c_stats[4]],
                                name=group_name,
                                marker_color=self.highlight2,
                                boxpoints=False,
                                showlegend=False,
                            ),
                            row=1, col=1
                        )
                     
                # author line 
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[c_value_author, c_value_author],
                        mode='lines',
                        line=dict(color=self.highlight1, width=4),
                        name=f'Author: {c_value_author:.1f}',
                        showlegend=False,
                        hovertemplate=f'Author: {c_value_author:.1f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # author rank (1,2)
                rank = author_metrics.get(f'rank{suffix}', 'N/A')
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=rank if isinstance(rank, (int, float)) else 0,
                        number={'font': {'color': self.highlight1, 'size': 60},
                                'valueformat': '.0f' },
                        title={
                            'text': f"<b>Glob. Rank of<br>{author_name}</b>",
                            'font': {'color': self.highlight1, 'size': 12}
                        }
                    ),
                    row=1, col=3
                )
           
                
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=n2, 
                        number={'font': {'color': self.highlight2, 'size': 60},
                                'valueformat': '.0f' },
                        title={
                            'text': f"<b>Number of authors in<br>{group_name[:20]}...</b>" if len(group_name) > 20 else f"<b>Number of authors in<br>{group_name}</b>",
                            'font': {'color': self.highlight2, 'size': 12}
                        }
                    ),
                    row=1, col=5
                )
                
                # ==================================================================
                # LINES 2 :  6 metrics
                # ==================================================================
                
                positions = [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6)]
                
                for i in range(6):
                    base_metric = metrics_base[i]
                    metric_with_suffix = metrics_with_suffix[i]
                    author_value = author_metrics.get(metric_with_suffix, 0)
                    
                    row, col = positions[i]
                    
                    # Box plot 
                    if base_metric in group_stats:
                        stats = group_stats[base_metric]
                        if isinstance(stats, list) and len(stats) >= 5:
                            fig.add_trace(
                                go.Box(
                                    q1=[stats[1]],
                                    median=[stats[2]],
                                    q3=[stats[3]],
                                    lowerfence=[stats[0]],
                                    upperfence=[stats[4]],
                                    name=group_name,
                                    marker_color=self.highlight2,
                                    boxpoints=False,
                                    showlegend=False,
                                    
                                ),
                                row=row, col=col
                            )
                    
                    # author line 
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[author_value, author_value],
                            mode='lines',
                            line=dict(color=self.highlight1, width=3),
                            name=f'Author: {author_value:.1f}',
                            showlegend=False,
                            hovertemplate=f'Author: {author_value:.1f}<extra></extra>'
                        ),
                        row=row, col=col
                    )
                
                # ==================================================================
                # Formating
                # ==================================================================
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor=self.bgc,
                    paper_bgcolor=self.bgc,
                    font={'color': self.lightAccent1},
                    showlegend=False,
                    # title={
                    #     'text': f"Comparison: {author_name} vs {group_name}",
                    #     'font': {'size': 20, 'color': self.lightAccent1},
                    #     'x': 0.5
                    # },
                    #margin={'l': 40, 'r': 40, 't': 100, 'b': 40}
                )
                
                # Axes
                boxplot_positions = [
                    (1,1),  # Composite score
                    (2,1), (2,2), (2,3), (2,4), (2,5), (2,6)  # 6 metrics
                ]

                for (row, col) in boxplot_positions:
                    fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
                    fig.update_yaxes(showgrid=True, gridcolor=self.darkAccent2, row=row, col=col)
               
                
                # annotation initialization
                annotations = []
                
                # Add C Score annotation
                annotations.append(dict(
                    x=0.5, y=c_value_author,
                    xref='x', yref='y',
                    text=f'Author: {c_value_author:.1f}',
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=self.highlight1,
                    ax=-40, ay=-30,
                    font=dict(size=10, color=self.highlight1),
                    xanchor='left'
                ))
                
                # Add metrics annotations
                x_positions = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                y_values = [author_metrics.get(metrics_with_suffix[i], 0) for i in range(6)]
                x_refs = ['x2', 'x3', 'x4', 'x5', 'x6', 'x7']
                y_refs = ['y2', 'y3', 'y4', 'y5', 'y6', 'y7']
                
                for i in range(6):
                    annotations.append(dict(
                        x=x_positions[i], y=y_values[i],
                        xref=x_refs[i], yref=y_refs[i],
                        text=f'Author: {y_values[i]:.0f}',
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=self.highlight1,
                        ax=-30, ay=-20,
                        font=dict(size=9, color=self.highlight1),
                        xanchor='left'
                    ))
                
                # Subtitiles
                
                annotations.append(dict(
                    x=0.08, y=1.07,
                    xref='paper', yref='paper',
                    text='<b>Composite Score (C)</b>',
                    showarrow=False,
                    font=dict(size=12, color=self.lightAccent1),
                    xanchor='center'
                ))
                
    
                metric_positions = [
                    (0.05, 0.52), (0.21, 0.52), (0.39, 0.52), (0.56, 0.52), (0.75, 0.52), (0.92, 0.52) 
                ]
                
                for i, (x, y) in enumerate(metric_positions):
                    annotations.append(dict(
                        x=x, y=y,
                        xref='paper', yref='paper',
                        text=f'<b>{metric_titles[i]}</b>',
                        showarrow=False,
                        font=dict(size=11, color=self.lightAccent1),
                        xanchor='center'
                    ))
                
                fig.update_layout(annotations=annotations)
                
                return fig
                
            except Exception as e:
                print(f"❌ Error creating comparison: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        def update_comparison(b=None):
            with output:
                clear_output(wait=True)
                
                author_name = author_search.value
                career_author = career_single_author.value == 'Career'
                year_val = year_author.value
                group_type_val = group_type.value
                group_name = group_selection.value
                exclude_self_val = exclude_self.value
                log_tf = log_transform.value
                
                if not author_name or not group_name:
                    print("❌ Please select both an author and a group")
                    return
                
                #print(f" Generating comparison for {author_name} vs {group_name}...")
                
                try:
                    fig = create_comparison_figures_author_group(
                        author_name, career_author, year_val, group_type_val,
                        group_name, exclude_self_val, log_tf
                    )
                    
                    if fig:
                        fig.show()
                    else:
                        print("❌ No data available")
                        
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        update_button.on_click(update_comparison)
        
        # ==========================================================================================
        # LAYOUT FINAL
        # ==========================================================================================
        
        controls = widgets.VBox([
            widgets.HBox([author_search, search_status, career_single_author, year_author]),
            widgets.HBox([group_type, group_selection, group_status]),
            widgets.HBox([exclude_self, log_transform, update_button])
        ])
        
        display(controls)
        display(output)
        
        update_group_options(None)
        
        update_comparison()

    #group_vs_group_layout
    def interactive_group_vs_group_comparison(self):
        """Interactive interface to compare two groups using box plots and ranks."""
        
        # ==========================================================================================
        # CONTROL WIDGETS - GROUP 1
        # ==========================================================================================
        
        group_type_1 = widgets.Dropdown(
            options=[
                ('Country', 'cntry'),
                ('Field', 'sm-field'), 
                ('Institution', 'inst_name')
            ],
            value='sm-field',
            description='Group1:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        group_selection_1 = widgets.Combobox(
            value='Clinical Medicine',
            placeholder='Select group...',
            options=[],
            description='Select1:',
            ensure_option=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        group_status_1 = widgets.HTML(value='')
        
        # ==========================================================================================
        # CONTROL WIDGETS - GROUP 2
        # ==========================================================================================
        
        group_type_2 = widgets.Dropdown(
            options=[
                ('Country', 'cntry'),
                ('Field', 'sm-field'), 
                ('Institution', 'inst_name')
            ],
            value='cntry',
            description='Group2:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        group_selection_2 = widgets.Combobox(
            value='United States',
            placeholder='Select group...',
            options=[],
            description='Select2:',
            ensure_option=False,
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        group_status_2 = widgets.HTML(value='')
        
        # ==========================================================================================
        # GENERAL OPTIONS 
        # ==========================================================================================
        
        career_single_group = widgets.Dropdown(
            options=['Career', 'Single Year'],
            value='Career',
            description='Dataset:',
            style={'description_width': '100px'}
        )
        
        year_group = widgets.Dropdown(
            options=['2017', '2018', '2019', '2020', '2021'],
            value='2020',
            description='Year:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='150px')
        )
        
        exclude_self = widgets.Checkbox(
            value=False,
            description='Exclude self-citations',
            style={'description_width': '100px'}
        )
        
        log_transform = widgets.Checkbox(
            value=False,
            description='Log transformed',
            style={'description_width': '150px'}
        )
        
        update_button = widgets.Button(
            description='Generate Comparison',
            button_style='success',
            icon='refresh',
            layout=widgets.Layout(width='250px')
        )
        
        output = widgets.Output()
        
        # ==========================================================================================
        # UPDATE FUNCTION 
        # ==========================================================================================
        
        def update_group_options(change, group_num):
          
            try:
                career = career_single_group.value == 'Career'
                year = year_group.value
                group = group_type_1.value if group_num == 1 else group_type_2.value
                
 
                if group == 'cntry':
                    api_type = 'country'
                elif group == 'sm-field':
                    api_type = 'field'
                else:  # inst_name
                    api_type = 'institution'
                
                
                url = f"{self.base_url}/aggregate/{api_type}"
                params = {'limit': 500}
                
                if not career:
                    params['year'] = year
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                results = data.get('results', [])
                
                
                options = []
                for result in results:
                    if group == 'cntry':
                        code = result.get('cntry', '').lower()
                        if code and code not in ['csk', 'nan']:
                            try:
                                name = coco.convert(code, to='name_short')
                                if name:
                                    options.append(name)
                            except:
                                pass
                    elif group == 'sm-field':
                        field = result.get('sm-field', '')
                        if field and field != 'Nan':
                            options.append(field)
                    else:  # inst_name
                        inst = result.get('inst_name', '')
                        if inst and inst != 'Nan':
                            options.append(inst)
                
                if group_num == 1:
                    group_selection_1.options = sorted(set(options))
                    group_status_1.value = f'<i>✓ {len(options)} groups available</i>'
                else:
                    group_selection_2.options = sorted(set(options))
                    group_status_2.value = f'<i>✓ {len(options)} groups available</i>'
                
            except Exception as e:
                if group_num == 1:
                    group_status_1.value = f'<i style="color:red">Error: {str(e)}</i>'
                else:
                    group_status_2.value = f'<i style="color:red">Error: {str(e)}</i>'
        
        # Observers
        group_type_1.observe(lambda change: update_group_options(change, 1), names='value')
        group_type_2.observe(lambda change: update_group_options(change, 2), names='value')
        career_single_group.observe(lambda change: [update_group_options(change, 1), update_group_options(change, 2)], names='value')
        year_group.observe(lambda change: [update_group_options(change, 1), update_group_options(change, 2)], names='value')
        
        # ==========================================================================================
        # MAIN FUNCTION
        # ==========================================================================================
        
        def create_comparison_figures_group_group(group_type_1_val, group_name_1, 
                                                group_type_2_val, group_name_2,
                                                career_group, year_val, 
                                                exclude_self_val, log_transform_val):
          
            
            try:
                prefix = 'career' if career_group else 'singleyr'
                key = f"{prefix}_{year_val}"
                
                # Group 1
                group_data_1, count_1 = self.get_aggregate_data(group_type_1_val, group_name_1, prefix)
                if group_data_1 is None or key not in group_data_1:
                    print(f"❌ Données non disponibles pour {group_name_1}")
                    return None
                group_stats_1 = group_data_1[key]
                
                try:
                    n1 = group_stats_1['c'][5]  
                except:
                    n1 = 0
                
                # Group 2
                group_data_2, count_2 = self.get_aggregate_data(group_type_2_val, group_name_2, prefix)
                if group_data_2 is None or key not in group_data_2:
                    print(f"❌ Données non disponibles pour {group_name_2}")
                    return None
                group_stats_2 = group_data_2[key]
                
                try:
                    n2 = group_stats_2['c'][5]  
                except:
                    n2 = 0
                
                suffix = ' (ns)' if exclude_self_val else ''
                metrics_base = ['nc', 'h', 'hm', 'ncs', 'ncsf', 'ncsfl', 'c']
                
                metric_titles = [
                    'NC', 
                    'H', 
                    'Hm', 
                    'NCS', 
                    'NCSF', 
                    'NCSFL', 
                    'Composite score (C)'
                ]
                
                fig = make_subplots(
                    rows=2,
                    cols=6,
                    specs=[
                        [
                            {'type': 'box'}, None,
                            {'type': 'indicator'}, None,
                            {'type': 'indicator'}, None
                        ],
                        [
                            {'type': 'box'}, {'type': 'box'}, {'type': 'box'},
                            {'type': 'box'}, {'type': 'box'}, {'type': 'box'}
                        ]
                    ],
                    row_heights=[0.45, 0.45],
                    horizontal_spacing=0.05,
                    vertical_spacing=0.15,
                    column_widths=[0.16]*6,
                )
                
                # =======================================================================================
                # LINE 1: Score C (1,1) + number of authors group 1 (1,3) + number of authors group2 (1,5)
                # =======================================================================================
                
                # Score C - Box plots 
                c_metric = metrics_base[6]
                
                # Group 1
                if c_metric in group_stats_1:
                    c_stats_1 = group_stats_1[c_metric]
                    if isinstance(c_stats_1, list) and len(c_stats_1) >= 5:
                        fig.add_trace(
                            go.Box(
                                q1=[c_stats_1[1]],
                                median=[c_stats_1[2]],
                                q3=[c_stats_1[3]],
                                lowerfence=[c_stats_1[0]],
                                upperfence=[c_stats_1[4]],
                                name=group_name_1,
                                marker_color=self.highlight1,
                                boxpoints=False,
                                showlegend=True,
                                legendgroup='group1',
                                offsetgroup='group1'
                            ),
                            row=1, col=1
                        )
                
                # Group 2
                if c_metric in group_stats_2:
                    c_stats_2 = group_stats_2[c_metric]
                    if isinstance(c_stats_2, list) and len(c_stats_2) >= 5:
                        fig.add_trace(
                            go.Box(
                                q1=[c_stats_2[1]],
                                median=[c_stats_2[2]],
                                q3=[c_stats_2[3]],
                                lowerfence=[c_stats_2[0]],
                                upperfence=[c_stats_2[4]],
                                name=group_name_2,
                                marker_color=self.highlight2,
                                boxpoints=False,
                                showlegend=True,
                                legendgroup='group2',
                                offsetgroup='group2'
                            ),
                            row=1, col=1
                        )
                
                # Number of author group 1 (1,3)
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=n1,
                        number={'font': {'color': self.highlight1, 'size': 60},
                                'valueformat': '.0f' },
                        title={
                             'text': f"<b>Number of Authors in<br>{group_name_1}</b>", 
                            'font': {'color': self.highlight1, 'size': 12}
                        }
                    ),
                    row=1, col=3
                )
                
                # Number of author group 1 2 (1,5)
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=n2,
                        number={'font': {'color': self.highlight2, 'size': 60},
                                'valueformat': '.0f' },
                        title={
                            'text': f"<b>Number of Authors in <br>{group_name_2}</b>", 
                            'font': {'color': self.highlight2, 'size': 12}
                        }
                    ),
                    row=1, col=5
                )
                
                # ==================================================================
                # LINE 2: 6 metrics
                # ==================================================================
                
                positions = [(2,1), (2,2), (2,3), (2,4), (2,5), (2,6)]
                
                for i in range(6):
                    base_metric = metrics_base[i]
                    row, col = positions[i]
                    
                    # Box plot group 1
                    if base_metric in group_stats_1:
                        stats_1 = group_stats_1[base_metric]
                        if isinstance(stats_1, list) and len(stats_1) >= 5:
                            fig.add_trace(
                                go.Box(
                                    q1=[stats_1[1]],
                                    median=[stats_1[2]],
                                    q3=[stats_1[3]],
                                    lowerfence=[stats_1[0]],
                                    upperfence=[stats_1[4]],
                                    name=group_name_1,
                                    marker_color=self.highlight1,
                                    boxpoints=False,
                                    showlegend=False,
                                    legendgroup='group1',
                                    offsetgroup='group1'
                                ),
                                row=row, col=col
                            )
                    
                    # Box plot group 2
                    if base_metric in group_stats_2:
                        stats_2 = group_stats_2[base_metric]
                        if isinstance(stats_2, list) and len(stats_2) >= 5:
                            fig.add_trace(
                                go.Box(
                                    q1=[stats_2[1]],
                                    median=[stats_2[2]],
                                    q3=[stats_2[3]],
                                    lowerfence=[stats_2[0]],
                                    upperfence=[stats_2[4]],
                                    name=group_name_2,
                                    marker_color=self.highlight2,
                                    boxpoints=False,
                                    showlegend=False,
                                    legendgroup='group2',
                                    offsetgroup='group2'
                                ),
                                row=row, col=col
                            )
                
                # ==================================================================
                # Formating
                # ==================================================================
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor=self.bgc,
                    paper_bgcolor=self.bgc,
                    font={'color': self.lightAccent1},
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.10,
                        xanchor="center",
                        x=0.5
                    ),
                    # title={
                    #     'text': f"Comparison: {group_name_1} vs {group_name_2}",
                    #     'font': {'size': 20, 'color': self.lightAccent1},
                    #     'x': 0.5
                    # },
                    boxmode='group'
                )
                
                # Axes
                boxplot_positions = [
                    (1,1),  # Composite score
                    (2,1), (2,2), (2,3), (2,4), (2,5), (2,6)  # 6 metrics
                ]
                
                for (row, col) in boxplot_positions:
                    fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
                    fig.update_yaxes(showgrid=True, gridcolor=self.darkAccent2, row=row, col=col)
                
                # Subtitiles
                annotations = []
                
                annotations.append(dict(
                    x=0.08, y=1.07,
                    xref='paper', yref='paper',
                    text='<b>Composite Score (C)</b>',
                    showarrow=False,
                    font=dict(size=12, color=self.lightAccent1),
                    xanchor='center'
                ))
                
                metric_positions = [
                    (0.05, 0.52), (0.21, 0.52), (0.39, 0.52), (0.56, 0.52), (0.75, 0.52), (0.92, 0.52)
                ]
                
                for i, (x, y) in enumerate(metric_positions):
                    annotations.append(dict(
                        x=x, y=y,
                        xref='paper', yref='paper',
                        text=f'<b>{metric_titles[i]}</b>',
                        showarrow=False,
                        font=dict(size=11, color=self.lightAccent1),
                        xanchor='center'
                    ))
                
                fig.update_layout(annotations=annotations)
                
                return fig
                
            except Exception as e:
                print(f"❌ Error creating comparison: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        def update_comparison(b=None):
            with output:
                clear_output(wait=True)
                
                group_type_1_val = group_type_1.value
                group_name_1 = group_selection_1.value
                group_type_2_val = group_type_2.value
                group_name_2 = group_selection_2.value
                career_group = career_single_group.value == 'Career'
                year_val = year_group.value
                exclude_self_val = exclude_self.value
                log_tf = log_transform.value
                
                if not group_name_1 or not group_name_2:
                    print("❌ Please select both groups")
                    return
                
                try:
                    fig = create_comparison_figures_group_group(
                        group_type_1_val, group_name_1,
                        group_type_2_val, group_name_2,
                        career_group, year_val,
                        exclude_self_val, log_tf
                    )
                    
                    if fig:
                        fig.show()
                    else:
                        print("❌ No data available")
                        
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        update_button.on_click(update_comparison)
        
        # ==========================================================================================
        # LAYOUT FINAL
        # ==========================================================================================
        
        controls = widgets.VBox([
            widgets.HBox([group_type_1, group_selection_1, group_status_1]),
            widgets.HBox([group_type_2, group_selection_2, group_status_2]),
            widgets.HBox([career_single_group, year_group, exclude_self, log_transform, update_button])
        ])
        
        display(controls)
        display(output)
        
        update_group_options(None, 1)
        update_group_options(None, 2)
        
        update_comparison()

    # ========================================================================
    # VISUALISATION - MAPS
    # ========================================================================

    def get_country_aggregates(self, year: str = "2021", career: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch aggregated data by country via the API.

        Args:
            year: Year (2017–2021)
            career: True for career data, False for single-year data
        Returns:
            A DataFrame containing statistics per country, or None if unavailable
        """
        try:
            url = f"{self.base_url}/aggregate/country"
            params = {'limit': 500}
            
            if not career:
                params['year'] = year
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print("❌ No aggregated data found.")
                return None
            
            country_data = []
            prefix = 'career' if career else 'singleyr'
            
            for result in results:
                country_code = result.get('cntry', '').lower()
                if not country_code or country_code in ['csk', 'nan']:
                    continue
                
                if country_code == 'sux':
                    country_name = 'Russia'
                elif country_code == 'ant':
                    country_name = 'Netherlands'
                elif country_code == 'scg':
                    country_name = 'Czech Republic'
                else:
                    country_name = coco.convert(country_code, to='name_short')
                
                agg_data = result.get('data')
                if isinstance(agg_data, str):
                    try:
                        agg_data = self.base64_decode_and_decompress(agg_data)
                    except Exception as e:
                        print(f" Decompression error for {country_code}: {e}")
                        continue
                
    
                year_key = f'{prefix}_{year}'
                if year_key not in agg_data:
                    continue
                
                year_data = agg_data[year_key]
                
                country_data.append({
                    'CODE': country_code,
                    'COUNTRY': country_name,
                    'data': year_data
                })

            if not country_data:
                print("❌ No valid data after processing")
                return None
            
            return pd.DataFrame(country_data)
        
        except Exception as e:
            print(f"❌ Error retrieving country aggregates: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_choropleth_map(self, metric: str = 'nc', statistic: str = 'median',
                        year: str = "2021", career: bool = True,
                        exclude_self_citations: bool = False) -> Optional[go.Figure]:
        """
        Create a world choropleth map for a given metric.
        """
        try:
            df = self.get_country_aggregates(year, career)
            
            if df is None or df.empty:
                print("❌ No data available")
                return None
            
            # Create a progress bar 
            total_countries = len(df)
            progress_bar = tqdm(total=100, desc=" Generating map", bar_format='{l_bar}{bar} {percentage:3.0f}%')
            
            metric_key = metric if not exclude_self_citations else metric
            
            stat_values = []
            valid_countries = []
            
            for idx, row in enumerate(df.iterrows()):
                _, row_data = row
                year_data = row_data['data']
                
                if metric_key in year_data:
                    metric_data = year_data[metric_key]
                    
                    if isinstance(metric_data, list) and len(metric_data) >= 5:
                        if statistic == 'min':
                            value = metric_data[0]
                        elif statistic == '25%':
                            value = metric_data[1]
                        elif statistic == 'median':
                            value = metric_data[2]
                        elif statistic == '75%':
                            value = metric_data[3]
                        elif statistic == 'max':
                            value = metric_data[4]
                        else:
                            value = metric_data[2]
                        
                        stat_values.append(value)
                        valid_countries.append(row_data['CODE'])
                
                # update progress bar
                progress = ((idx + 1) / total_countries) * 50
                progress_bar.n = progress
                progress_bar.refresh()
            
            if not stat_values:
                progress_bar.close()
                print("❌ No valid statistic extracted.")
                return None
            
            country_names = []

            for idx, code in enumerate(valid_countries):
                if code == 'sux':
                    country_names.append('Russia')
                elif code == 'ant':
                    country_names.append('Netherlands')
                elif code == 'scg':
                    country_names.append('Czech Republic')
                else:
                    try:
                        country_name = coco.convert(code, to='name_short')
                        country_names.append(country_name)
                    except:
                        country_names.append('Unknown')
                
                
                progress = 50 + ((idx + 1) / len(valid_countries)) * 40
                progress_bar.n = progress
                progress_bar.refresh()
            
    
            map_df = pd.DataFrame({
                'CODE': [c.upper() for c in valid_countries],  
                'COUNTRY': country_names,
                'value': stat_values
            })

            map_df['metric'] = metric
            map_df['metric_name'] = self._get_metric_long_name(career, metric, 0, False)

            # Create the map
            fig = go.Figure(data=go.Choropleth(
                locations=map_df['CODE'],
                locationmode='ISO-3',
                z=map_df['value'],
                text=map_df['COUNTRY'],
                colorscale='Viridis',
                customdata=map_df[['metric', 'metric_name']],
                autocolorscale=False,
                reversescale=False,
                marker_line_color=self.darkAccent3,
                marker_line_width=0.5,
                colorbar=dict(
                    title=dict(
                        text=statistic.title(),
                        font=dict(color=self.lightAccent1)
                    ),
                    tickfont=dict(color=self.lightAccent1),
                    thickness=15,
                    len=0.7
                ),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'metric=%{customdata[0]}<br>'
                    'code=%{location}<br>'
                    'country=%{text}<br>'
                    'metric_name=%{customdata[1]}<br>'
                    f'{statistic.title()}: %{{z:.2f}}<br>'
                    '<extra></extra>'
                )
            ))
            
            fig.update_geos(
                showcountries=True,
                countrycolor=self.darkAccent3,
                showcoastlines=True,
                coastlinecolor=self.darkAccent3,
                projection_type='natural earth',
                bgcolor=self.bgc
            )
            
            fig.update_layout(
                geo=dict(bgcolor=self.bgc),
                plot_bgcolor=self.bgc,
                paper_bgcolor=self.bgc,
                font=dict(color=self.lightAccent1),
                height=600,
                coloraxis_colorbar=dict(
                    title=dict(
                        text=statistic.title(),
                        font=dict(color=self.lightAccent1)
                    ),
                    tickfont=dict(color=self.lightAccent1)
                )
            )
            
            progress_bar.n = 100
            progress_bar.refresh()
            progress_bar.close()
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating map: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def interactive_choropleth_map(self):
        """
        Interactive interface to explore choropleth maps by country.
        Reproduces the server dashboard interface for geo maps.
        """
        # ==========================================================================================
        # CONTROL WIDGETS
        # ==========================================================================================
        
        # Data type
        dataset_type = widgets.Dropdown(
            options=['Career', 'Single Year'],
            value='Career',
            description='Dataset:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='250px')
        )
        
        # Year selected
        year_selector = widgets.Dropdown(
            options=['2017', '2018', '2019', '2020', '2021'],
            value='2021',
            description='Year:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px')
        )
        
        # Metric
        metric_selector = widgets.Dropdown(
            options=[
                ('Number of Citations', 'nc'),
                ('H-index', 'h'),
                ('Hm-index', 'hm'),
                ('Cit. Single Authored', 'ncs'),
                ('Cit. Single+First', 'ncsf'),
                ('Cit. Single+First+Last', 'ncsfl'),
                ('Composite Score', 'c')
            ],
            value='h',
            description='Metric:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='300px')
        )
        
        # Statistic
        statistic_selector = widgets.Dropdown(
            options=[
                ('25th Percentile', '25%'),
                ('75th Percentile', '75%'),
                ('Minimum', 'min'),
                ('Median', 'median'),
                ('Maximum', 'max')   
            ],
            value='median',
            description='Statistic:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='250px')
        )
        
        # Options
        exclude_self = widgets.Checkbox(
            value=False,
            description='Exclude self-citations',
            style={'description_width': '100px'}
        )
        
        # Update button
        update_button = widgets.Button(
            description='Generate Map',
            button_style='success',
            icon='globe',
            layout=widgets.Layout(width='200px')
        )
        
        output = widgets.Output()
        
        # ==========================================================================================
        # UPDATE FUNCTION
        # ==========================================================================================
        
        def update_map(b=None):
            with output:        
                clear_output(wait=True)
                career = dataset_type.value == 'Career'
                year = year_selector.value
                metric = metric_selector.value
                statistic = statistic_selector.value
                exclude = exclude_self.value
                
                # print(f" Generating map for {metric} ({statistic}) - {year}...")
                
                try:
                    fig = self.create_choropleth_map(
                        metric=metric,
                        statistic=statistic,
                        year=year,
                        career=career,
                        exclude_self_citations=exclude
                    )
                    
                    clear_output(wait=True)
                    if fig:
                        fig.show()
                    else:
                        print("❌ Unable to generate map")
                
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        update_button.on_click(update_map)
        
        # ==========================================================================================
        # LAYOUT FINAL
        # ==========================================================================================
        
        controls = widgets.VBox([
            widgets.HBox([dataset_type, year_selector, metric_selector]),
            widgets.HBox([statistic_selector, exclude_self, update_button]),
        ])
        
        display(controls)
        display(output)
        update_map()
   
   

    # ========================================================================
    # MAPS WITHOUT PROGRESS BAR
    # ========================================================================

    # def create_choropleth_map(self, metric: str = 'nc', statistic: str = 'median',
    #                         year: str = "2021", career: bool = True,
    #                         exclude_self_citations: bool = False) -> Optional[go.Figure]:
    #     """
    #     Create a world choropleth map for a given metric.

    #     Args:
    #         metric: Base metric ('nc', 'h', 'hm', 'ncs', 'ncsf', 'ncsfl', 'c')
    #         statistic: Statistic to display ('count', 'median', 'min', 'max', '25%', '75%')
    #         year: Year (2017-2021)
    #         career: True for career-long, False for single-year
    #         exclude_self_citations: Exclude self-citations
            
    #     Returns:
    #         Plotly Figure or None
    #     """
    #     try:

    #         df = self.get_country_aggregates(year, career)
            
    #         if df is None or df.empty:
    #             print("❌ No data available")
    #             return None
            
    #         #Add the suffix for self-citations if needed.
    #         metric_key = metric if not exclude_self_citations else metric
            
    #         # Extract the statistics for the metric
    #         stat_values = []
    #         valid_countries = []
            
            
    #         for idx, row in df.iterrows():
    #             year_data = row['data']
                
    #             if metric_key not in year_data:
    #                 continue
                
    #             metric_data = year_data[metric_key]
                
    #             # Structure: [min, Q1, median, Q3, max].
    #             if not isinstance(metric_data, list) or len(metric_data) < 5:
    #                 continue
                
    #             #Map the requested statistic.
    #             if statistic == 'min':
    #                 value = metric_data[0]
    #             elif statistic == '25%':
    #                 value = metric_data[1]
    #             elif statistic == 'median':
    #                 value = metric_data[2]
    #             elif statistic == '75%':
    #                 value = metric_data[3]
    #             elif statistic == 'max':
    #                 value = metric_data[4]
    #             else:
    #                 value = metric_data[2]  # Defaut: median
                
    #             stat_values.append(value)
    #             valid_countries.append(row['CODE'])
            
    #         if not stat_values:
    #             print("❌ No valid statistic extracted.")
    #             return None
            
    #         #Create the DataFrame for the map with country names.
    #         country_names = []
    #         for code in valid_countries:
    #             if code == 'sux':
    #                 country_names.append('Russia')
    #             elif code == 'ant':
    #                 country_names.append('Netherlands')
    #             elif code == 'scg':
    #                 country_names.append('Czech Republic')
    #             else:
    #                 country_names.append(coco.convert(code, to='name_short'))
            
    #         map_df = pd.DataFrame({
    #             'CODE': [c.upper() for c in valid_countries],  
    #             'COUNTRY': country_names,
    #             'value': stat_values
    #         })
            
    #         # Layout
    #         prefix_text = 'Career-long' if career else 'Single-year'
    #         metric_name = self._get_metric_long_name(career, metric, 0, False)

    #         map_df['metric'] = metric
    #         map_df['metric_name'] = metric_name

    #         # Create the map using Plotly graph_objects for more control.
    #         fig = go.Figure(data=go.Choropleth(
    #             locations=map_df['CODE'],
    #             locationmode='ISO-3',
    #             z=map_df['value'],
    #             text=map_df['COUNTRY'],
    #             colorscale='Viridis',
    #             customdata=map_df[['metric', 'metric_name']],
    #             autocolorscale=False,
    #             reversescale=False,
    #             marker_line_color=self.darkAccent3,
    #             marker_line_width=0.5,
    #             colorbar=dict(
    #                 title=dict(
    #                     text=statistic.title(),
    #                     font=dict(color=self.lightAccent1)
    #                 ),
    #                 tickfont=dict(color=self.lightAccent1),
    #                 thickness=15,
    #                 len=0.7
    #             ),
    #             hovertemplate='<b>%{text}</b><br>' 'metric=%{customdata[0]}<br>' +
    #                         'code=%{location}<br>' +
    #                         'country=%{text}<br>' +
    #                         'metric_name=%{customdata[1]}<br>' +
    #                         f'{statistic.title()}: %{{z:.2f}}<br>' +
    #                         '<extra></extra>'
    #         ))
            
    #         fig.update_geos(
    #             showcountries=True,
    #             countrycolor=self.darkAccent3,
    #             showcoastlines=True,
    #             coastlinecolor=self.darkAccent3,
    #             projection_type='natural earth',
    #             bgcolor=self.bgc
    #         )
            
    #         fig.update_layout(
    #             # title={
    #             #     'text': f"{prefix_text} {metric_name} - {statistic.title()} by Country ({year})",
    #             #     'font': {'size': 18, 'color': self.lightAccent1},
    #             #     'x': 0.5
    #             # },
    #             geo=dict(bgcolor=self.bgc),
    #             plot_bgcolor=self.bgc,
    #             paper_bgcolor=self.bgc,
    #             font=dict(color=self.lightAccent1),
    #             height=600,
    #             coloraxis_colorbar=dict(
    #                 title=dict(
    #                     text=statistic.title(),
    #                     font=dict(color=self.lightAccent1)
    #                 ),
    #                 tickfont=dict(color=self.lightAccent1)
    #             )
    #         )
            
    #         return fig
        
    #     except Exception as e:
    #         print(f"❌ Error creating map: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None

    # def interactive_choropleth_map(self):
    #     """
    #     Interactive interface to explore choropleth maps by country.
    #     Reproduces the server dashboard interface for geo maps.
    #     """
    #     # ==========================================================================================
    #     # CONTROL WIDGETS
    #     # ==========================================================================================
        
    #     # Data type
    #     dataset_type = widgets.Dropdown(
    #         options=['Career', 'Single Year'],
    #         value='Career',
    #         description='Dataset:',
    #         style={'description_width': '100px'},
    #         layout=widgets.Layout(width='250px')
    #     )
        
    #     # Year selected
    #     year_selector = widgets.Dropdown(
    #         options=['2017', '2018', '2019', '2020', '2021'],
    #         value='2021',
    #         description='Year:',
    #         style={'description_width': '100px'},
    #         layout=widgets.Layout(width='200px')
    #     )
        
    #     # Metric
    #     metric_selector = widgets.Dropdown(
    #         options=[
    #             ('Number of Citations', 'nc'),
    #             ('H-index', 'h'),
    #             ('Hm-index', 'hm'),
    #             ('Cit. Single Authored', 'ncs'),
    #             ('Cit. Single+First', 'ncsf'),
    #             ('Cit. Single+First+Last', 'ncsfl'),
    #             ('Composite Score', 'c')
    #         ],
    #         value='h',
    #         description='Metric:',
    #         style={'description_width': '100px'},
    #         layout=widgets.Layout(width='300px')
    #     )
        
    #     # Statistic
    #     statistic_selector = widgets.Dropdown(
    #         options=[
    #             ('25th Percentile', '25%'),
    #             ('75th Percentile', '75%'),
    #             ('Minimum', 'min'),
    #             ('Median', 'median'),
    #             ('Maximum', 'max')   
    #         ],
    #         value='median',
    #         description='Statistic:',
    #         style={'description_width': '100px'},
    #         layout=widgets.Layout(width='250px')
    #     )
        
    #     # Options
    #     exclude_self = widgets.Checkbox(
    #         value=False,
    #         description='Exclude self-citations',
    #         style={'description_width': '100px'}
    #     )
        
    #     # Update button
    #     update_button = widgets.Button(
    #         description='Generate Map',
    #         button_style='success',
    #         icon='globe',
    #         layout=widgets.Layout(width='200px')
    #     )
        
    #     output = widgets.Output()
        
    #     # ==========================================================================================
    #     # UPDATE DUNCTION
    #     # ==========================================================================================
        
    #     def update_map(b=None):
    #         with output:        
    #             career = dataset_type.value == 'Career'
    #             year = year_selector.value
    #             metric = metric_selector.value
    #             statistic = statistic_selector.value
    #             exclude = exclude_self.value
                
    #             print(f" Generating map for {metric} ({statistic})...")
                
    #             try:
    #                 fig = self.create_choropleth_map(
    #                     metric=metric,
    #                     statistic=statistic,
    #                     year=year,
    #                     career=career,
    #                     exclude_self_citations=exclude
    #                 )
                    
    #                 if fig:
    #                     clear_output(wait=True)
    #                     fig.show()
                        
    #                 else:
    #                     print("❌ Unable to generate map")
                
    #             except Exception as e:
    #                 print(f"❌ Error: {str(e)}")
    #                 import traceback
    #                 traceback.print_exc()
        
    #     update_button.on_click(update_map)
        
    #     # ==========================================================================================
    #     # LAYOUT FINAL
    #     # ==========================================================================================
        
    #     controls = widgets.VBox([
    #         widgets.HBox([dataset_type, year_selector, metric_selector]),
    #         widgets.HBox([statistic_selector, exclude_self, update_button]),

    #     ])
        
    #     display(controls)
    #     display(output)
    #     update_map()



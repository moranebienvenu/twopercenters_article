---
numbering:
  heading_2: false
  figure:
    template: Fig. %s
tags:
  - reproducible publishing
  - interactive articles
  - reproducible research
  - open access
date: 2025-11-24
---

+++ { "part": "abstract" }
We built an interactive dashboard to explore citation data from the standardized citation indicators dataset covering 100,000 top-cited researchers across 22 disciplines. The system uses Elasticsearch for search and Plotly Dash for visualization, allowing users to compare researchers, institutions, countries, and fields through various bibliometric metrics. Users can toggle between career-long and single-year data, exclude self-citations, and examine temporal trends. The platform runs on open-source infrastructure and provides an alternative interface for bibliometric analysis. All source code and deployment instructions are fully open, so anyone can modify the platform for new datasets or expand its functionality.
+++

# Introduction

To date, citations to scholarly articles have served as the primary currency for attributing credibility to their authors. From the academic job market to research funding decisions, citation counts and related metrics remain central indicators shaping outcomes. Google Scholar, a citation-based search engine, has become the dominant gateway to explore articles, author profiles and evaluative bibliography, largely because it is free and user-friendly [](https://doi.org/10.2139/ssrn.2921021).

However, its influence is not without problems. Unlike curated databases such as Web of Science or Scopus, Google Scholar operates with minimal quality control as it relies on [web scraping](https://en.wikipedia.org/wiki/Web_scraping). It automatically indexes a broad range of sources, including non-peer-reviewed manuscripts, conference abstracts, institutional repositories, and even duplicated or erroneous records. The same mechanism makes the platform vulnerable to manipulation, as recently demonstrated in a striking case where a fabricated researcher amassed 380 citations and an h-index of 19 after uploading just 20 ChatGPT-generated articles across three preprint servers (https://doi.org/10.1038/s41598-025-88709-7). Another well-known example is Larry the Cat [](10.1126/science.zl99qni
) who famously dethroned [Felis Domesticus Chester Willard](https://www.science.org/content/article/cat-co-authored-influential-physics-paper) to become the world's most cited cat by using a citation-boosting service advertised on Facebook.

Nevertheless, in today’s digital environment, Google Scholar confronts a challenge far more consequential than the whimsical phenomenon of feline citation races: the growing reliance on AI-driven tools for scholarly search [](https://doi.org/10.1090/noti2838). A recent [blog post](https://hannahshelley.neocities.org/blog/2025_08_13_GoogleScholar) by Hannah Shelly powerfully captures how Scholar service is nearing the [Google Graveyard](https://killedbygoogle.com) for this exact reason and highlights the levels of unpreparedness by the academic community:

> "academia has built critical infrastructure around a free commercial service with zero guarantees."

The coming years will show whether Google Scholar follows the fate of Microsoft Academic. More critically, the task ahead is to establish trustworthy platforms that engage the scholarly community through a robust bibliometric database, especially as literature search and discovery increasingly migrate to AI systems. One of the most notable efforts in this direction is the science-wide author databases of standardized citation indicators, exemplified by the work of [](https://doi.org/10.1371/journal.pbio.3000384) and further expanded by large-scale bibliometric platforms such as [OpenAlex](https://openalex.org), which now provides an even broader and more comprehensive mapping of the global research landscape.

First released in 2019, this dataset encompasses the 100,000 most cited authors across 22 scientific disciplines and 174 subfields, as indexed by Scopus. It provides rankings for both a given year and across an author’s entire career. The primary ranking metric is a composite citation indicator (c-score), which integrates multiple citation-based measures to provide a more balanced assessment of research impact rather than bare citation counts or h-index. As of today, seven versions of this dataset are available in tabular format for anyone to explore freely, as well as the source code to replicate the calculation of the metrics on Databricks using PySpark.

{button}`Elsevier Data Repository<https://elsevier.digitalcommonsdata.com/datasets/btchxktzyw/7>`

Here, we aimed at demonstrating how community-hosted open-source tools can provide robust alternatives to commercial bibliometric platforms. Specifically, we developed an interactive dashboard that transforms complex tabular citation data into an accessible platform with advanced search capabilities and multiple analytical perspectives for exploring and comparing research impact metrics across authors, institutions, countries, and scientific fields.

# Methods

We designed and implemented a data processing and visualization pipeline to transform the standardized citation indicators dataset into an interactive web-based dashboard. Our approach prioritized open-source tools, reproducible infrastructure, and scalable deployment to demonstrate how community-hosted platforms can effectively serve large-scale bibliometric data. 

The diagram below illustrates the complete system architecture, from raw data ingestion through to user-facing analytical interfaces.

::::{figure}
:label: chart
:::{mermaid}
graph TD
    A[Raw Citation Data<br/>Science-wide Author Database] --> B[Data Preprocessing Pipeline]
    B --> C[Data Compression & Encoding]
    C --> D[Elasticsearch Indexing]

    D --> E[Individual Researcher Index<br/>Author profiles, metrics, affiliations]
    D --> F[Aggregation Index<br/>Country, field, institutional summaries]

    E --> G[Elasticsearch Cluster<br/>Distributed Search & Analytics]
    F --> G

    G --> H[Plotly Dash Web Application]

    H --> I[Interactive World Map<br/>Global citation distributions]
    H --> J[Author Search Interface<br/>Autocomplete capabilities]
    H --> K[Comparison Tools<br/>Author vs Author/Group]
    H --> L[Temporal Analysis<br/>Metric evolution over time]

    M[NeuroLibre Dokku Platform] --> N[Elasticsearch Deployment<br/>Security & access controls]
    M --> O[Dashboard Deployment<br/>Scalable hosting]

    N --> G
    O --> H

    style A fill:#e1f5fe
    style G fill:#fff3e0
    style H fill:#f3e5f5
    style M fill:#e8f5e8
:::
Schematic overview of the interactive data application system architecture.
::::


## Data Processing and Indexing Pipeline

The dashboard infrastructure was built upon a comprehensive data processing pipeline that transforms the standardized citation indicators dataset into an interactive visualization platform. The implementation consists of three primary components: data preprocessing, Elasticsearch indexing, and web-based dashboard development.

### Data Preprocessing and Aggregation

Raw data from the science-wide author databases underwent systematic preprocessing to prepare multiple analytical perspectives. The pipeline generated composite datasets organized by individual researchers (career-long and single-year metrics) and aggregate views grouped by country, scientific field, and institutional affiliation. Data compression and encoding techniques were implemented to optimize storage and retrieval performance within the search infrastructure.

### Elasticsearch Integration

The processed datasets were indexed using Elasticsearch, a distributed search and analytics engine, to enable rapid querying and filtering capabilities essential for interactive visualization. Two primary indexing functions were developed to handle different data structures:

The main indexing function creates document mappings for individual researcher profiles, storing author names, country affiliations, institutional names, scientific fields, temporal coverage, and compressed data objects. Search metadata including country, institution, and field classifications are extracted from the most recent year of available data to ensure current relevance.

A specialized aggregation indexing function handles country, field, and institutional summary statistics, creating streamlined indices optimized for aggregate-level queries and visualizations.

The indexing process utilizes parallel bulk operations to efficiently process large datasets while maintaining data integrity through error handling and index refresh operations. Document compression using base64 encoding minimizes storage requirements while preserving complete dataset accessibility.

### Dashboard Architecture

The interactive dashboard was developed using Plotly Dash, a Python web application framework optimized for analytical visualizations. To that effect, the architecture employed a page-based routing system where each analytical component was implemented as a separate module, facilitating navigation between different analytical perspectives

The primary landing page provided a comprehensive overview interface featuring an interactive world map visualization built with Plotly Express. This interface allowed users to explore global distributions of citation metrics with dynamic filtering capabilities. Users could switch between career-long and single-year datasets, select different years (2017-2021), and examine various summary statistics (minimum, maximum, median, 25th and 75th percentiles).

Interactive callback functions handled real-time user interactions, including map clicks that triggered detailed country-specific views displaying institutional and researcher breakdowns. The interface integrated collapsible accordion sections containing specialized comparison tools for different analytical perspectives:

- **Author search functionality**: Implemented through dedicated search interfaces with autocomplete capabilities
- **Author vs. author comparisons**: Side-by-side metric comparisons with interactive visualizations
- **Author vs. group comparisons**: Individual researcher performance against aggregate statistics
- **Group vs. group comparisons**: Comparative analysis between countries, institutions, or fields
- **Temporal trend analysis**: Individual researcher metric evolution over time

The dashboard connected directly to the Elasticsearch cluster through authenticated connections, retrieving and processing data in real-time. Custom utility functions handled Elasticsearch queries, data decompression, and result formatting. The application employed efficient pagination and scrolling mechanisms for large result sets, ensuring responsive performance even with extensive datasets.

Visual styling was implemented using a dark theme with custom color palettes and responsive design components, such as loading spinners to enhance user experience during data retrieval operations.

### Deployment Infrastructure

Both the Elasticsearch instance and dashboard application were deployed on NeuroLibre's Dokku platform, providing scalable hosting with automated deployment capabilities. The Elasticsearch cluster was configured with appropriate security measures and access controls, while the dashboard application interfaced with the search backend through authenticated connections.

Environment-specific configurations enabled deployment flexibility, with support for both local development and production environments. The implementation included proper error handling, logging, and monitoring capabilities to ensure reliable operation in production settings.

All source code, configuration files, and deployment scripts were made freely available through the project repository at https://github.com/Notebook-Factory/twopercenters, enabling reproducibility and community contributions to the platform.

# Results

:::{attention} Enable Computational Interactivity 

**<span style="color:red">To enable interactivity, attach a runtime by clicking the `⏻` icon in the top-right corner of the figure 2 panel</span>**. If no runtime is attached, the figure will remain in its default state: hover and basic interactions are available.

**<span style="color:red">Once loaded, three buttons appear in the corner. Click the middle play button `▶️` to activate figure 2.</span>** When the static figure is replaced by an interactive Plotly chart, use the toggles to modify the display. Use <span style="display:inline-block; transform: scaleX(-1) scaleY(-1);">↪️</span> to revert to the original static figure.

:::

:::{figure} #fig1cell
:label: fig1
:name: maps
:placeholder: ./static/fig1.png

Global distribution of research excellence metrics (snapshot displays median h-index for career-long data from 2021) among top 2% scientists by country, as visualized on a world map.
***Users can select the type of analysis (Career or Single Year), choose the year (2017–2021), set the statistic analysis (25%, 75%. median, min, max) or choose the metric (h, nc, hm, ncs, ncsf, ncsfl).***

:::

The default state of [](#fig1) displays median H-index values across countries, derived from career-span academic performance data of the most cited researchers worldwide in 2021. Color scale represents median H-index from 0 (purple) to 70+ (yellow), with darker regions indicating lower citation impact and brighter regions showing higher research productivity. This page enables exploration of all the metrics that go into the calculation of the c-score (dropdown), as well as the selection of different statistics (dropdown). This panel also allows exploration of country-specific institutional rankings when clicked on the respective country, determined by the affiliation of the researchers who made the cut.

---
::::{seealso} Click here to interact with the dashboard inline
:class: dropdown
:::{iframe} https://twopercenters.db.neurolibre.org/
:width: 100%
:::
::::

{button}`Click here to open the dashboard in a new tab<https://twopercenters.db.neurolibre.org/>`

---

:::{note} To activate Figure 3 interactivity: **<span style="color:red">click the play button `▶️`</span>** if available. If only the `⏻` icon is present, click it first to start the runtime, then click `▶️`.
:::

:::{figure} #fig2cell
:label: fig2
:name: individual_gauge
:placeholder: ./static/fig2.png

Individual researcher performance panel showing example bibliometric analysis for John P.A. Ioannidis. (a) Summary metrics panel displaying composite score (5.19) and key bibliometric indicators including total citations (88.6K), H-index (132), and HM-index (66.9), with corresponding percentile rankings and citation distributions across single-authored and collaborative works.
***Users can enter another author’s name (a minimum of three letters is required), and a list of available authors will be shown. Users can also select the type of analysis (Career or Single Year), choose the year (2017–2021), set limits (by country, field, or institute), and toggle the exclusion of self-citations.***

:::

:::{note} To activate Figure 4 interactivity: **<span style="color:red">click the play button `▶️`</span>** if available. If only the `⏻` icon is present, click it first to start the runtime, then click `▶️`.
:::

:::{figure} #fig3cell
:label: fig3
:name: individual_career_single_year
:placeholder: ./static/fig3.png

Temporal trends comparing career-long versus single-year performance trajectories from 2017-2021, showing rank progression, composite scores, total citations, and H-index evolution for both assessment periods
***Users can enter another author’s name (a minimum of three letters is required), and a list of available authors will be shown. Users can also toggle the exclusion of self-citations.***
:::

[](#fig2) and [](#fig3) present a multifaceted research impact assessment through both aggregate career metrics and year-specific performance indicators. A key feature includes the option to exclude self-citations, allowing users to examine how researcher rankings change when self-referential patterns are removed from the analysis. The temporal comparison between career-long and single-year data reveals how annual fluctuations in citation patterns contrast with the stability of cumulative metrics. This dual-perspective approach enables more objective evaluation of researcher trajectories, particularly valuable for understanding sustained versus ephemeral impact in scholarly productivity assessments.

:::{note} To activate Figure 5 interactivity: **<span style="color:red">click the play button `▶️`</span>** if available. If only the `⏻` icon is present, click it first to start the runtime, then click `▶️`.
:::

:::{figure} #fig4cell
:label: fig4
:name: author_vs_author_layout
:placeholder: ./static/fig4.png

Author-vs-author comparison between researchers showing composite scores and citation distributions. Abbreviations: Number of Citations (NC), H-index (H), Hm-index (Hm), Number of citations to single-authored papers (NCS), Number of citations to single and first-authored papers (NCSF), Number of citations to single, first, and last-authored papers (NCSFL). 
***Users can enter another authors' names (a minimum of three letters is required), and a list of available authors will be shown. Users can also select the type of analysis (Career or Single Year), choose the year (2017–2021), and toggle the exclusion of self-citations or Log transformation.***
:::

:::{note} To activate Figure 6 interactivity: **<span style="color:red">click the play button `▶️`</span>** if available. If only the `⏻` icon is present, click it first to start the runtime, then click `▶️`.
:::

:::{figure} #fig5cell
:label: fig5
:name: author_vs_group_layout
:placeholder: ./static/fig5.png

Author-vs-group analysis displaying individual performance against field averages (41,350 authors in Clinical Medicine).Abbreviations: Number of Citations (NC), H-index (H), Hm-index (Hm), Number of citations to single-authored papers (NCS), Number of citations to single and first-authored papers (NCSF), Number of citations to single, first, and last-authored papers (NCSFL). ***Users can enter another author’s name (minimum of three letters) to see a list of available authors. They can also select the type of analysis (Career or Single Year) and choose the year (2017–2021) for the single author. For the group comparator, users can select the organisation (by country, field, or institute), and a list of available options will be shown. Additionally, users can toggle the exclusion of self-citations or apply a log transformation.***

:::

:::{note} To activate Figure 7 interactivity: **<span style="color:red">click the play button `▶️`</span>** if available. If only the `⏻` icon is present, click it first to start the runtime, then click `▶️`.
:::

:::{figure} #fig6cell
:label: fig6
:name: group_vs_group_layout
:placeholder: ./static/fig6.png

Group-vs-group comparison between research fields and countries with statistical distributions. Abbreviations: Number of Citations (NC), H-index (H), Hm-index (Hm), Number of citations to single-authored papers (NCS), Number of citations to single and first-authored papers (NCSF), Number of citations to single, first, and last-authored papers (NCSFL). 
***For each group, users can select the organisation (by country, field, institute or All), and a list of available options will be shown. They can also select the type of analysis (Career or Single Year), choose the year (2017–2021),toggle the exclusion of self-citations or apply a log transformation.***

:::

[](#fig4), [](#fig5) and [](#fig6) demonstrate comparison options across multiple organizational levels, from individual researchers to entire research fields and national systems. The panels allow users to toggle between career-span and single-year metrics while maintaining the option to exclude self-citations for more objective assessments. Statistical visualizations include box plots and distribution analyses that reveal performance variability within groups, providing context for individual achievements relative to peer cohorts. This multi-scale approach supports evidence-based evaluation of research impact across institutional, national, and disciplinary boundaries

# Conclusion

The dashboard presented herein converts complex citation datasets into an interactive platform that enables researchers to systematically examine impact metrics across disciplines, perform institutional comparisons, and monitor career trajectories. The platform is entirely free of advertisements, paywalls, and opaque algorithms. Furthermore, all components are open source and reproducible, thereby facilitating extension, adaptation to specific research domains, and independent deployment by the broader community.

The infrastructure described herein (see [](#chart)) serves as a pragmatic and effective model for the development of community-driven scholarly tools. While it does not introduce fundamentally novel technologies, its strength lies in empowering users to maintain autonomy over their data and analytical resources. This approach underscores the value of transparent, self-managed solutions in advancing open and reproducible research practices.

In light of the transformative impact of AI on research discovery and consumption, the availability of transparent, community-governed alternatives is increasingly essential. Future developments will focus on augmenting the dashboard with the most recent dataset releases and expanding the platform’s functionality to include a programmatic API. This enhancement will facilitate direct access to citation metrics from researchers’ scripts and applications, thereby further democratizing bibliometric data and fostering the creation of custom analytical tools grounded in open infrastructure.

+++ { "part": "acknowledgement" }
Hosting was provided by NeuroLibre, a next-generation publication platform funded by the Canadian Open Neuroscience Platform, Brain Canada, Quebec Bioimaging Network, and Institut TransMedTech at Polytechnique Montréal.
+++

+++ { "part": "data_availability" }
The original dataset has been made available by [](https://doi.org/10.1371/journal.pbio.3000384) at https://elsevier.digitalcommonsdata.com/datasets/btchxktzyw/7. 
+++
---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Figure 1: Interactive NeuroTmap Analysis

```{code-cell} python
:tags: [thebe-init]
!pip install plotly ipywidgets requests pandas numpy

{code-cell}
:tags: [thebe-active]
:label: fig1-interactive

from Dash_client import DashNeuroTmapClient

# Initialiser le client
client = DashNeuroTmapClient()

# Créer l'interface interactive
client.create_advanced_interface()
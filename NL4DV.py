from nl4dv import NL4DV
import altair as alt
# from altair import vega, vegalite
from IPython.display import display

data_url="./Datasets/owid-energy-data-2.csv"
label_attribute = None
dependency_parser_config = {"name": "spacy", "model": "en_core_web_sm", "parser": None}

nl4dv_instance = NL4DV(verbose=False, 
                       debug=True, 
                       data_url=data_url, 
                       label_attribute=label_attribute, 
                       dependency_parser_config=dependency_parser_config
                       )
nl4dv_response = nl4dv_instance.analyze_query("Compare the US' GDP and China's.")
print(nl4dv_response)
# display(alt.display.html_renderer(nl4dv_response['visList'][0]['vlSpec']), raw=True)
from lida.modules import Manager
import lida
import pandas   as pd

import os
os.environ["LIDA_ALLOW_CODE_EVAL"] = "1"

datapath = "../Datasets/housing.csv"
lida = Manager()
summary = lida.summarize(datapath) # generate data summary

goals = lida.generate_goals(summary, n=5) # generate goals

# generate code specifications for charts
vis_specs = lida.generate_viz(summary=summary, goal=goals[0], library="matplotlib") # altair, matplotlib etc

# execute code to return charts (raster images or other formats)
charts = lida.execute_viz(code_specs=vis_specs, data=pd.read_csv(datapath), summary=summary)
charts[0].code.show(    )   # show code
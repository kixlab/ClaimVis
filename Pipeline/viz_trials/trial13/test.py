from lida.modules import Manager
from lida.datamodel import Goal
import lida
import pandas   as pd
import base64
from io import BytesIO
from PIL import Image
import os

# save subtable to csv
data_url=f"sub_table.csv"
sub_table = pd.read_csv(data_url)
vis_task = "Show a bar chart representing the IMDB rating of each movie."

# perform trial
os.environ["LIDA_ALLOW_CODE_EVAL"] = "1"
lida = Manager()

summary = lida.summarize(data_url)
goal = Goal(
            question=vis_task + ". Sample only 30 random datapoints when there are over 100 datapoints.",
            index=0,
            visualization="",
            rationale=""
        )

# generate code specifications for charts
vis_specs = lida.generate_viz(summary=summary, goal=goal, library="seaborn") # altair, matplotlib etc
# execute code to return charts (raster images or other formats)
charts = lida.execute_viz(code_specs=vis_specs, data=pd.read_csv(data_url), summary=summary)
print(charts[0].code, "\n**********")

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

decoded_image = decode_base64_to_image(charts[0].raster)
decoded_image.show()
# Save the image to trial{idx}.png
decoded_image.save(f"trial13.png")
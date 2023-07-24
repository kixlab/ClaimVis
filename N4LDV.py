from nl4dv import NL4DV
nl4dv_instance = NL4DV(data_url="./Datasets/colleges.csv")
response = nl4dv_instance.analyze_query("Show debt for New England.")

print(response)
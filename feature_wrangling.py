import io
import json
import random

import pandas as pd
import torch

expression_dict = {}
for data_set in ["test", "train", "val"]:
    with open(f"generated_features/{data_set}/{data_set}_expression.pt", "rb") as f:
        buffer = io.BytesIO(f.read())
        expression_dict[data_set] = torch.load(buffer, map_location=torch.device("cpu"))
expression_results = [
    pic for inner_list in expression_dict.values() for pic in inner_list
]

object_dict = {}
for data_set in ["test", "train", "val"]:
    with open(
        f"generated_features/{data_set}/{data_set}_object_detection.pt", "rb"
    ) as f:
        buffer = io.BytesIO(f.read())
        object_dict[data_set] = torch.load(buffer, map_location=torch.device("cpu"))
object_results = [obj for inner_list in object_dict.values() for obj in inner_list]

places_dict = {}
for data_set in ["test", "train", "val"]:
    with open(
        f"generated_features/{data_set}/{data_set}_places365.json", "r"
    ) as json_file:
        places_dict[data_set] = json.load(json_file)
places_result = {
    a: b for inner_dict in places_dict.values() for (a, b) in inner_dict.items()
}

# Get expression classes
with open("class_files/expression_classes.txt", "r") as f:
    expression_classes = [s.strip() for s in f.readlines()]

# Get coco classes
with open("class_files/coco_classes.txt", "r") as f:
    coco_classes = [s.strip() for s in f.readlines()]

# Get places365 classes
file_name = "class_files/places365_classes.txt"
places365_classes = list()
with open(file_name) as class_file:
    for line in class_file:
        places365_classes.append(line.strip().split(" ")[0][3:])

# Load the existing data
main_data = pd.read_csv("data/data.csv")
val_data = main_data[main_data.dataset == "validation"]

# Wrangle the expression data
dict_res = {}
for mini_dict in expression_results:
    key = list(mini_dict.keys())[0]
    faces = mini_dict[key]
    faces = list(mini_dict.values())[0]
    res = {class_name: [] for class_name in expression_classes}
    res["likely"] = []
    for face in faces:
        res.update({key: value + [0] for key, value in res.items()})
        face = list(face.values())[0]
        face_expression = face["classes"]
        probabiliites = face["probs"]
        for expression, probability in zip(face_expression, probabiliites):
            res[expression_classes[int(expression)]][-1] = float(probability.numpy())
        res["likely"][-1] = expression_classes[
            int(face_expression[probabiliites.argmax()])
        ]

    dict_res[key] = res

# Convert to a dataframe and rename columns
expression_data = pd.DataFrame.from_dict(dict_res, orient="index")
expression_data.rename(columns=lambda x: f"expression_{x}", inplace=True)

# Join the two dataframes
data_with_expression = pd.merge(
    val_data, expression_data, left_on="id", right_index=True, how="left"
)

# Drop the "N/A" objects
object_class_list = [item for item in coco_classes if item != "N/A"]

# Wrangle the object data
object_dictionary = {}
for photo in object_results:
    photo_id = photo["image"].split(".")[0]
    photo_dictionary = {obj: [] for obj in object_class_list}
    x = photo["output"]
    for obj, prob in zip(x["labels"], x["scores"]):
        photo_dictionary[coco_classes[obj]].append(float(prob))
    object_dictionary[photo_id] = photo_dictionary

# Convert to DataFrame, drop `__background__` column and rename rest
object_data = pd.DataFrame.from_dict(object_dictionary, orient="index")
object_data.drop("__background__", axis=1, inplace=True)
object_data.rename(columns=lambda col: f"object_{col}", inplace=True)

# Join the two dataframes
data_with_object = pd.merge(
    data_with_expression, object_data, left_on="id", right_index=True, how="left"
)

# Wrangle the places data
places_dictionary = {}
for photo_id, photo in places_result.items():
    photo_dictionary = {place: 0 for place in places365_classes}
    for place, prob in zip(photo["class"], photo["prob"]):
        photo_dictionary[place] = prob
    places_dictionary[photo_id] = photo_dictionary

# Convert to DataFrame and rename columns
places_data = pd.DataFrame.from_dict(places_dictionary, orient="index")
places_data.rename(columns=lambda col: f"place_{col}", inplace=True)

# Join the dataframes
all_data = pd.merge(
    data_with_object,
    places_data,
    left_on="id",
    right_index=True,
    how="left",
)

# Write data to file
all_data.to_json("data/data_with_features.json")
all_data.to_csv("data/data_with_features.csv", index=False)

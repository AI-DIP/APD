import matplotlib.pyplot as plt
import pandas as pd
from adversarial_prototype_decomposition.classifier import classifiers as apd

def get_proto_info(model:apd.APD_ClassifierScaler, columns, scaler=None):
    if scaler is None:
        proto_info = pd.DataFrame(model.scaler.inverse_transform(model.proto_ensemble_.proto),columns = columns)
    else:
        proto_info = pd.DataFrame(scaler.inverse_transform(model.proto_ensemble_.proto),columns = columns)
    proto_info["Class"] = model.proto_ensemble_.proto_labels
    used = []
    for pair in model.region_stats["Pair"]:
        p = model.proto_ensemble_.unpairCantor(pair)
        used.extend(p)
    proto_info["Times Paired"] = proto_info.index.map(pd.Series(used).value_counts()).fillna(0).astype(int)
    return proto_info

def get_region_info(model:apd.APD_Classifier):
    region_info = pd.DataFrame(model.region_stats)
    l = list(zip(*region_info["Pair"].apply(model.proto_ensemble_.unpairCantor)))
    region_info["ProtoClass0"] = l[0]
    region_info["ProtoClass1"] = l[1]
    region_info.columns = ["Pair","Class_0_size","Class_1_size","Rank","ProtoClass0","ProtoClass1"]
    return region_info
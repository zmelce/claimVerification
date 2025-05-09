import pandas as pd
import ast
from collections import Counter
import csv

dfs = pd.read_excel("ULKB_UnifiedMapping_V1.xlsx", sheet_name=None)
dfs2= dfs["UM_RAW_V1"]
mapList = dfs2.mapping
verbset= dfs2["Propbank roleset"]

prop_verb = pd.DataFrame(columns=['verb_propbank','verbnet'])

verb_propbank=[]
verbnet=[]
for i, verb in enumerate(verbset):
    verb = verb.replace(".","-")
    elem = mapList[i].replace("; ", ",")
    entities = ast.literal_eval(elem)
    for key, value in entities.items():
        if 'vnArg' in value:
            verbnet_item = value['vnArg']
            propbank = verb+":"+key
            verbnet.append(verbnet_item.lower())
            verb_propbank.append(propbank)


print(Counter(verbnet))



data_df = pd.DataFrame(columns=['publisher', 'review_date', 'claim_text', 'label', 'review_url','review_headline','source','relation','target'])

rel_count=0
prob_verbnet_map=[]
amr_count=0
arg0_v=[]
arg1_v=[]
arg2_v=[]
arg3_v=[]
arg4_v=[]
arg5_v=[]
with open('claimReview_oneSent_AMR_10k.csv', 'r') as file:
    csvreader = csv.reader(file,delimiter =';')
    header = next(csvreader)
    for i, row in enumerate(csvreader):
        data_df.at[i, 'publisher'] = row[0]
        data_df.at[i, 'review_date'] = row[1]
        data_df.at[i, 'claim_text'] = row[2]
        data_df.at[i, 'label'] = row[3]
        data_df.at[i, 'review_url'] = row[4]
        data_df.at[i, 'review_headline'] = row[5]

        if row[6] != 'NAN':
            amr_count +=1
            source_list =ast.literal_eval(row[6])
            rel_list = ast.literal_eval(row[7])
            data_df.at[i, 'source'] = source_list
            data_df.at[i, 'target'] = ast.literal_eval(row[8])
            for j, item in enumerate(rel_list):
                rel_count+=1
                new_item= source_list[j] +item
                if new_item in verb_propbank:
                    ind = verb_propbank.index(new_item)
                    verbnet_rol = verbnet[ind]
                    prob_verbnet = item + "/" + verbnet_rol #
                    rel_list[j] = prob_verbnet #
                    #rel_list[j] = verbnet_rol
                    #prob_verbnet_map.append(source_list[j] +prob_verbnet)
                    prob_verbnet_map.append(source_list[j] + verbnet_rol)
                    if (item == ":ARG0"):
                        arg0_v.append(verbnet_rol)
                    elif (item == ":ARG1"):
                        arg1_v.append(verbnet_rol)
                    elif (item == ":ARG2"):
                        arg2_v.append(verbnet_rol)
                    elif (item == ":ARG3"):
                        arg3_v.append(verbnet_rol)
                    elif (item == ":ARG4"):
                        arg4_v.append(verbnet_rol)
                    elif (item == ":ARG5"):
                        arg5_v.append(verbnet_rol)

            data_df.at[i, 'relation'] =rel_list


            # data_df.at[i, 'source'] = ast.literal_eval(row[6].replace('\"', ''))
            # data_df.at[i, 'relation'] = ast.literal_eval(row[7].replace('\"', ''))
            # data_df.at[i, 'target'] = ast.literal_eval(row[8].replace('\"', ''))
print("mm")

print("claimReview mapping ", Counter(prob_verbnet_map))
print("AMR graph count: ", amr_count)
data_df.to_csv('claimReview_oneSent_AMR_VerbNet_Prop_10k.csv', header=True, index=False, encoding='utf-8')
#Propbank roleset

print("arg0 map_size: ",len(arg0_v))
print("arg0 verbnet_roles: ", list(set(arg0_v)))
print("arg0, ", Counter(arg0_v))
print("arg0 roles, ", Counter(arg0_v).keys())

print("arg1 map_size: ",len(arg1_v))
print("arg1 verbnet_roles: ", list(set(arg1_v)))
print("arg1, ", Counter(arg1_v))
print("arg0 roles, ", Counter(arg1_v).keys())


print("arg2 map_size: ",len(arg2_v))
print("arg2 verbnet_roles: ", list(set(arg2_v)))
print("arg2, ", Counter(arg2_v))
print("arg2 roles, ", Counter(arg2_v).keys())


print("arg3 map_size: ",len(arg3_v))
print("arg3 verbnet_roles: ", list(set(arg3_v)))
print("arg3, ", Counter(arg3_v))
print("arg3 roles, ", Counter(arg3_v).keys())


print("arg4 map_size: ",len(arg4_v))
print("arg4 verbnet_roles: ", list(set(arg4_v)))
print("arg4, ", Counter(arg4_v))
print("arg4 roles, ", Counter(arg4_v).keys())


print("arg5 map_size: ",len(arg5_v))
print("arg5 verbnet_roles: ", list(set(arg5_v)))
print("arg5, ", Counter(arg5_v))
print("arg5 roles, ", Counter(arg5_v).keys())


def get_pb_vn_mappings(self, verbnet):
    vn_members = [member.name for member in verbnet.get_members()]
    res = {}

    # Initial population of rolesets
    for roleset in self.rolesets:
        res[roleset] = {}
        if not self.rolesets[roleset].vnc:
            if roleset in self.pb2vn:
                print("roleset found in mapping file;;" + roleset + str(self.pb2vn[roleset]))
                self.rolesets[roleset].vnc = ["-".join(vnc.split("-")[1:]) for vnc in self.pb2vn[roleset]]
            elif roleset.split(".")[0] in vn_members:
                print ("no vn mapping, but it's in verbnet;;" + roleset)
        for vnc in self.rolesets[roleset].vnc:
            new_c = verbnet.find_correct_subclass(vnc, roleset.split(".")[0])
            if not new_c:
                print ("class not found;;" + roleset + ";;" + vnc)
            elif new_c == vnc:
                res[roleset][vnc] = {}
                print ("class is good;;" + roleset + ";;" + vnc)
            else:
                res[roleset][new_c] = {}
                print ("class/member found, update;;" + roleset + ";;" + vnc + ";;" + new_c)
        if len(res[roleset]):
            for arg in self.rolesets[roleset].role_mappings:
                for vnc in res[roleset]:
                    if vnc in self.rolesets[roleset].role_mappings[arg]:
                        res[roleset][vnc][arg] = self.rolesets[roleset].role_mappings[arg][vnc]

    # Filling in role mappings from identical roleset mappings
    # Implements the assumption that if two PB rolesets for the same verb map to the same class,
    # their role mappings will also be identical
    for roleset in res:
        for roleset2 in res:
            if roleset.split(".")[0] == roleset2.split(".")[0]:
                for mapping in res[roleset]:
                    if mapping in res[roleset2]:
                        for arg in res[roleset][mapping]:
                            res[roleset2][mapping][arg] = res[roleset][mapping][arg]

    # Removing empty mappings
    res = {r :res[r] for r in res if res[r] != {}}

    return res

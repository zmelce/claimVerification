import calendar
import re
from itertools import combinations
from nltk.stem import WordNetLemmatizer
import penman
from penman import constant
from penman.models.noop import NoOpModel
from deep_translator import GoogleTranslator
import pandas as pd

def translator(source_lang, target_lang, x_language_token):
    translated_token = GoogleTranslator(source=source_lang, target=target_lang).translate(text=x_language_token)
    return translated_token

class AMR2Triples:

    lemmatizer = WordNetLemmatizer()
    op_pattern = re.compile("op([0-9]+)")
    ARG_REGEX = re.compile("ARG([0-9]+)")
    propbank_pattern = re.compile("([a-z0-9]+_)*(([a-z]+)-)+(\d\d)")
    non_core_roles = ['accompanier', 'age', 'beneficiary', 'concession', 'condition', 'consist-of', 'destination',
                      'direction', 'domain', 'duration', 'example', 'extent', 'frequency', 'instrument', 'location',
                      'manner', 'medium', 'mod', 'ord', 'part', 'path', 'prep-with', 'purpose', 'quant', 'source',
                      'subevent', 'time', 'topic', 'value']
    conjunctions = ['or', 'and']
    ignored_roles = ['name', ':instance', ':entities', ':entity', ':surface_form', ':type', ':uri']

    @classmethod
    def print_triples(cls, triples, title):
        #print('\n{}:\n'.format(title))
        df_triples = pd.DataFrame(columns=['source', 'relation', 'target'])
        sources=[]
        relations=[]
        targets=[]
        for source, source_id, relation, target, target_id in triples:
            #print('{}\t{}\t{}'.format(source, relation, target))
            sources.append(source)
            relations.append(relation)
            targets.append(target)
        df_triples.at[0,'source']=sources
        df_triples.at[0, 'relation'] =relations
        df_triples.at[0, 'target'] = targets
        #print('\n')
        return df_triples

    @classmethod
    def concat_name(cls, names_ops):
        name = ''
        for i in range(1, 15):
            if i in names_ops:
                name += ' ' + names_ops[i]
        return name

    @classmethod
    def get_variable_text(cls, var_id, names, dates):
        surface_text = ''
        if var_id in names:
            surface_text = names[var_id]
        elif var_id in dates:
            surface_text = dates[var_id]
        return surface_text.strip()

    @classmethod
    def get_triples(cls, graph):
        triples = graph.triples.copy()
        processed_triples = []

        name_var_list, var_to_type, var_to_name, name_to_var = list(), dict(), dict(), dict()
        grouped_names_ops, names = dict(), dict()
        mod_maps, mod_resolved = dict(), set()
        date_var_list, grouped_var_to_date, dates = list(), dict(), dict()
        ordinal_var_list, ordinal_value = list(), dict()
        temporal_var_list, temporal_value = list(), dict()
        and_var_list, and_values = list(), dict()

        for source, relation, target in triples:
            if relation == ':instance':
                var_to_type[source] = target
                if target == ':amr-unknown':
                    amr_unknown_var = source
                elif target == ':name':
                    name_var_list.append(source)
                elif target == ':time':
                    date_var_list.append(source)
                elif target == 'ordinal-entity':
                    ordinal_var_list.append(source)
                elif target == ':quant':
                    temporal_var_list.append(source)
                elif target == ':and':
                    and_var_list.append(source)

            elif relation == ':name':
                var_to_name[source] = target
            elif relation == ':mod' and target != 'amr-unknown':
                if target == 'expressive':
                    continue
                mod_values = mod_maps.get(source, list())
                mod_values.append(target)
                mod_maps[source] = mod_values
            elif relation in ['year', 'month', 'day', 'weekday'] and source in date_var_list:
                var_to_date_group = grouped_var_to_date.get(source, dict())
                var_to_date_group[relation] = target
                grouped_var_to_date[source] = var_to_date_group
            elif relation == ':value' and source in ordinal_var_list:
                ordinal_value[source] = target
            elif relation == ':quant' and source in temporal_var_list:
                temporal_value[source] = target
            # collecting all op* relations
            elif re.match(AMR2Triples.op_pattern, relation):
                if source in name_var_list:
                    op_pos = int(AMR2Triples.op_pattern.match(relation).group(1))
                    name_ops = grouped_names_ops.get(source, dict())
                    name_ops[op_pos] = str(target).replace("\"", "")
                    grouped_names_ops[source] = name_ops
                elif source in and_var_list:
                    and_ops = and_values.get(source, set())
                    and_ops.add(target)
                    and_values[source] = and_ops

        for var in var_to_name:
            name_to_var[var_to_name[var]] = var


        ##########################################
        for source, relation, target in triples:
            source_id = source
            target_id = target

            if target == 'interrogative':
                continue
            if relation in [':instance', ':entities', ':entity', ':id', ':type', ':surface_form',
                            ':uri'] or re.match(AMR2Triples.op_pattern, relation):
                # we have already processed these triples and collected the necessary information
                continue

            if relation == ':name':
                tempTriples = list(filter(lambda x: x[0] == target_id, triples))
                for s, r, t in tempTriples:
                    if r != ':instance':
                        target = t

            if str(var_to_type[source]) == 'name':
                continue

            if source in var_to_type:
                source = str(var_to_type[source])

            if target in dates:
                target = dates[target]

            if target in var_to_type:
                target = str(var_to_type[target])

            if target in ordinal_value:
                target = str(ordinal_value[target])

            if target in temporal_value:
                target = str(temporal_value[target])

            nsource= source.split('-')[0]
            ntarget= target.split('-')[0]
            processed_triples.append([source, source_id, relation, target, target_id])
            #processed_triples.append([translator('en','de',nsource), source_id, relation, translator('en','de',ntarget), target_id])
        df_triples = AMR2Triples.print_triples(processed_triples, 'Processed triples')
        return df_triples

    # @classmethod
    # def get_flat_triples(cls, sentence_text, penman_tree):
    #     triple_info = list()
    #     frame_args = dict()
    #     id_to_type = dict()
    #     reified_to_rel = dict()
    #     processed_triples, var_to_name, var_to_type, names, dates, ordinal_value, temporal_value, amr_unknown_var, top_node = \
    #         AMR2Triples.get_triples(sentence_text, penman_tree)
    #     for subject, source_id, relation, target, target_id in processed_triples:
    #         id_to_type[source_id] = subject
    #         id_to_type[target_id] = target
    #         subject_text, object_text = '', ''
    #         if source_id in names:
    #             subject_text = names[source_id]
    #         elif source_id in dates:
    #             subject_text = dates[source_id]
    #             id_to_type[source_id] = 'date-entity'
    #         if target_id in names:
    #             object_text = names[target_id]
    #         elif target_id in dates:
    #             object_text = dates[target_id]
    #             id_to_type[target_id] = 'date-entity'
    #         subject_text = subject_text.strip()
    #         object_text = object_text.strip()
    #
    #         # select subjects that are frames
    #         if re.match(AMR2Triples.propbank_pattern, str(subject)):
    #             if re.match(AMR2Triples.propbank_pattern, str(target)):
    #                 target = re.match(AMR2Triples.propbank_pattern, str(target)).group(2)
    #             # we have handled these before (and & or)
    #             if subject in AMR2Triples.conjunctions or target in AMR2Triples.conjunctions:
    #                 continue
    #             args = frame_args.get(source_id, dict())
    #             if re.match(AMR2Triples.ARG_REGEX, relation) or relation in AMR2Triples.non_core_roles:
    #                 args[relation] = target_id
    #             frame_args[source_id] = args
    #         elif relation not in AMR2Triples.ignored_roles and not re.match(AMR2Triples.ARG_REGEX, relation):
    #             subject_type = str(var_to_type[source_id]).split()[-1]
    #
    #             triple = dict()
    #             triple['subj_text'] = subject_text
    #             triple['subj_type'] = str(subject).strip()
    #             triple['subj_id'] = source_id
    #             triple['predicate'] = "{}.{}".format(str(subject_type).strip(), str(relation).strip())
    #             triple['predicate_id'] = source_id
    #             triple['obj_text'] = object_text
    #             triple['obj_type'] = str(target).strip()
    #             triple['obj_id'] = target_id
    #             triple['amr_unknown_var'] = amr_unknown_var
    #             triple_info.append(triple)
    #
    #     for frame_id in frame_args:
    #         frame_roles = frame_args[frame_id]
    #
    #         if id_to_type[frame_id] == 'have-rel-role-91':
    #             if 'ARG2' in frame_roles:
    #                 reified_to_rel[frame_id] = id_to_type[frame_roles['ARG2']]
    #             elif 'ARG3' in frame_roles:
    #                 reified_to_rel[frame_id] = id_to_type[frame_roles['ARG3']]
    #             else:
    #                 reified_to_rel[frame_id] = 'relation'
    #             if 'ARG0' in frame_roles and 'ARG1' in frame_roles:
    #                 for role in ['ARG2', 'ARG3']:
    #                     if role in frame_roles:
    #                         del frame_roles[role]
    #
    #         if id_to_type[frame_id] == 'have-org-role-91':
    #             if 'ARG2' in frame_roles:
    #                 reified_to_rel[frame_id] = id_to_type[frame_roles['ARG2']]
    #             elif 'ARG3' in frame_roles:
    #                 reified_to_rel[frame_id] = id_to_type[frame_roles['ARG3']]
    #             else:
    #                 reified_to_rel[frame_id] = 'position'
    #
    #             if 'ARG0' in frame_roles and 'ARG1' in frame_roles:
    #                 for role in ['ARG2', 'ARG3']:
    #                     if role in frame_roles:
    #                         del frame_roles[role]
    #
    #         # logic to handle the special case of frames with a single argument
    #         if len(frame_roles) == 1:
    #             frame_roles['unknown'] = "unknown"
    #             id_to_type['unknown'] = 'unknown'
    #
    #         rel_keys = sorted(list(frame_roles.keys()))
    #         for role1, role2 in combinations(rel_keys, 2):
    #             triple = dict()
    #             triple['subj_text'] = AMR2Triples.get_variable_text(frame_roles[role1], names, dates)
    #             triple['subj_type'] = str(id_to_type[frame_roles[role1]]).strip()
    #             triple['subj_id'] = frame_roles[role1]
    #             triple['predicate'] = '{}.{}.{}'.format(id_to_type[frame_id], role1.lower(), role2.lower())
    #             triple['predicate_id'] = frame_id
    #             triple['obj_text'] = AMR2Triples.get_variable_text(frame_roles[role2], names, dates)
    #             triple['obj_type'] = str(id_to_type[frame_roles[role2]]).strip()
    #             triple['obj_id'] = frame_roles[role2]
    #             triple['amr_unknown_var'] = amr_unknown_var
    #             triple_info.append(triple)
    #
    #     return triple_info, names, reified_to_rel, top_node

import amrlib

import csv

stog = amrlib.load_stog_model()
#sent1='This is the system trying.'
#graphs = stog.parse_sents(['This is the system trying.'])
#sent='Edward Snowden: Osama bin Laden is still alive living in the Bahamas.'
#sent='2 nurses who were in ITU in Swansea have died today.'
data_df = pd.DataFrame(columns=['publisher', 'review_date', 'claim_text', 'label', 'review_url','review_headline','source','relation','target'])

with open("claimReview_oneSent.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for i, row in enumerate(csvreader):
        claim= [row[2]]

        graphs = stog.parse_sents(claim)
        df_triples =pd.DataFrame(columns=['source', 'relation', 'target'])
        for temp in graphs:
            X = temp
            #print(X)
            Y = penman.decode(X,  model=NoOpModel())
            #print(Y)
            inst = Y.instances()
            edge = Y.edges()
            g2 = penman.Graph(Y.instances() + Y.edges() + Y.attributes())
            example = AMR2Triples()
            #triple_info, names, reified_to_rel, top_node =example.get_flat_triples(sent1, g2)
            df_triples = example.get_triples(g2)
            #processed_triples, var_to_name, var_to_type, names, dates, ordinal_value, temporal_value, amr_unknown_var, _= example.get_triples(sent1, g2)
            # print(triple_info)
            # print(names)
            # print(reified_to_rel)
            # print(top_node)
        data_df.at[i, 'publisher'] = row[0]
        data_df.at[i, 'review_date'] = row[1]
        data_df.at[i, 'claim_text'] = row[2]
        data_df.at[i, 'label'] = row[3]
        data_df.at[i, 'review_url'] = row[4]
        data_df.at[i, 'review_headline'] = row[5]
        data_df.at[i, 'source'] = df_triples.iloc[0]['source']
        data_df.at[i, 'relation'] = df_triples.iloc[0]['relation']
        #target =  df_triples.iloc[0]['target']
        #target = [s.replace('\\', '') for s in target]
        data_df.at[i, 'target'] = df_triples.iloc[0]['target']

data_df.to_csv('claimReview_oneSent_AMR.csv', header=True, index=False, encoding='utf-8')

import csv
import pandas as pd
import pickle
import ast
from genre.fairseq_model import mGENRE
from genre.trie import Trie, MarisaTrie

def get_wikidata_IDs(data_file):
  model = mGENRE.from_pretrained("../models/fairseq_multilingual_entity_disambiguation.tar.gz").eval()
  
  with open("../data/lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
     lang_title2wikidataID = pickle.load(f)
  
  with open("../data/titles_lang_all105_trie_with_redirect.pkl", "rb") as f:
     trie = Trie.load_from_dict(pickle.load(f))
  
  data_df = pd.DataFrame(columns=['publisher', 'review_date', 'claim_text', 'label', 'review_url','review_headline','tagged_claim','WikidataID'])
  results=[]
  with open(data_file, 'r') as file:
      csvreader = csv.reader(file, delimiter=',')
      header = next(csvreader)
      for i, row in enumerate(csvreader):
          data_df.at[i, 'publisher'] = row[0]
          data_df.at[i, 'review_date'] = row[7]
          data_df.at[i, 'claim_text'] = row[2]
          data_df.at[i, 'label'] = row[3]
          data_df.at[i, 'review_url'] = row[4]
          data_df.at[i, 'review_headline'] = row[5]
          #data_df.at[i, 'translated_claim'] = row[6]
          data_df.at[i, 'tagged_claim'] = row[6]
          if row[6] != 'NAN':
              tagClaim = ast.literal_eval(row[6])
              res = model.sample(
                                 tagClaim,
                                 prefix_allowed_tokens_fn=lambda batch_id, sent: [
                                     e for e in trie.get(sent.tolist())
                                     if e < len(model.task.target_dictionary)
                                     # for huggingface/transformers
                                     # if e < len(model2.tokenizer) - 1
                                 ],
                                 text_to_id=lambda x: max(lang_title2wikidataID[tuple(reversed(x.split(" >> ")))], key=lambda y: int(y[1:])),
                                 marginalize=False,
                             )
              data_df.at[i, 'WikidataID'] = res
          else :
  
              data_df.at[i, 'WikidataID'] = 'NAN'
  return data_df
  
def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        return requests.get(url, params=params)
    except:
        return 'There was and error'


def getTags(tagID):
    entities=[]
    for i in tagID:
        id = i
        params = {
            'action': 'wbgetentities',
            'ids': id,
            'format': 'json',
            'languages': 'en'
        }
        data = fetch_wikidata(params)
        if not isinstance(data, str):
            data = data.json()
            try:
                title = data['entities'][id]['labels']['en']['value']
                entities.append(title)
            except:
                print('not_found')
            #result = {'title':title}

    return entities


def getResult(id):
    params = {
        'action': 'wbgetentities',
        'ids': id,
        'format': 'json',
        'languages': 'en'
    }
    data = fetch_wikidata(params)
    data = data.json()

    try:
        title = data['entities'][id]['labels']['en']['value']
    except:
        title = 'not_found'
    try:
        alternate_names = [v['value'] for v in data['entities'][id]['aliases']['en']]
    except:
        alternate_names = 'not_found'
    try:
        description = data['entities'][id]['descriptions']['en']['value']
    except:
        description = 'not_found'
    try:
        instance_of = [v['mainsnak']['datavalue']['value']['id'] for v in data['entities'][id]['claims']['P31']]
    except:
        instance_of = 'not_found'
    try:
        part_of = [v['mainsnak']['datavalue']['value']['id'] for v in data['entities'][id]['claims']['P361']]
    except:
        part_of = 'not_found'
    try:
        founded_by = [v['mainsnak']['datavalue']['value']['numeric-id'] for v in data['entities'][id]['claims']['P112']]
    except:
        founded_by = 'not_found'
    try:
        nick_names = [v['mainsnak']['datavalue']['value']['text'] for v in data['entities'][id]['claims']['P1449']]
    except:
        nick_names = 'not_found'
    try:
        official_websites = [v['mainsnak']['datavalue']['value'] for v in data['entities'][id]['claims']['P856']]
    except:
        official_websites = 'not_found'
    try:
        categories = [v['mainsnak']['datavalue']['value']['numeric-id'] for v in data['entities'][id]['claims']['P910']]
    except:
        categories = 'not_found'
    try:
        inception = data['entities'][id]['claims']['P571'][0]['mainsnak']['datavalue']['value']['time']
    except:
        inception = 'not_found'

    try:
        employer = [v['mainsnak']['datavalue']['value']['id'] for v in data['entities'][id]['claims']['P108']]
    except:
        employer = 'not_found'

    try:
        member_of = [v['mainsnak']['datavalue']['value']['id'] for v in data['entities'][id]['claims']['P463']]
    except:
        member_of = 'not_found'

    try:
        position_held = [v['mainsnak']['datavalue']['value']['id'] for v in data['entities'][id]['claims']['P39']]
    except:
        position_held = 'not_found'

    instance_of = getTags(instance_of)
    employer = getTags(employer)
    member_of = getTags(member_of)
    position_held = getTags(position_held)
    part_of = getTags(part_of)


    result = {
        #'wikidata_id':id,
        'title':title,
        'description':description,
        'alternate_names':alternate_names,
        'instance_of':instance_of,
        'employer' : employer,
        'member_of': member_of,
        'position_held': position_held,
        'part_of':part_of
        #'founded_by':founded_by,
        #'main_categories':categories,
        }

    return result

def main(): 
    df_wikiIDs= get_wikidata_IDs(data_file)
    data_KGs = pd.DataFrame(columns=['publisher', 'claim_text', 'label', 'review_url','review_headline','wikiData'])
  
    for index, row in df_wikiIDs.iterrows():
        data_KGs.at[index, 'publisher'] = row[0]
        data_KGs.at[index, 'claim_text'] = row[2]
        data_KGs.at[index, 'label'] = row[3]
        data_KGs.at[index, 'review_url'] = row[4]
        data_KGs.at[index, 'review_headline'] = row[5]
        if row[7]!= 'NAN':
            wikiIDs= str(row[7])
            res = re.sub(r"tensor\(-?[0-9]+\.[0-9]+\)", "\'NAN\'", wikiIDs)
            entities = ast.literal_eval(res)
            idList=[]
            source=[]
            relation=[]
            target=[]
            temp=[]
            for entity in entities:
                id =entity[0]['id']
                idList.append(id)
                Y = getResult(id)
                temp.append(Y)
                data_KGs.at[index, 'wikiData'] = temp

        else:
            data_KGs.at[index, 'wikiData'] = 'NAN'

if __name__ == "__main__":
    main()








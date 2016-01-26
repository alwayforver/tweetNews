import bz2

file = "/home/jialu/Documents/dbpedia/freebase_links.nt"
dbpedia_freebase_links_dict = dict()

count = 0
with open(file, "r") as input:
    for line in input:
        try:
            count += 1
            if count % 100000 == 0:
                print str(count) + " links handled"
            elements = line.strip().split(' ')
            dbpedia_freebase_links_dict[elements[0][1:-1]] = elements[-2][1:-1]
        except:
            print line.strip()
            continue

import requests
import os
import json

doc_folder = '/home/jialu/Documents/eventcube/text_extracted/'
name_entity_file = '/home/jialu/Documents/eventcube/free_base_entities.txt'

covered_count = 0
missing_count = 0
with open(name_entity_file, 'w') as output:
    for filename in os.listdir(doc_folder):
        doc_id = int(filename[:filename.find('.txt')])
        output.write(str(doc_id) + '\t')

        with open(doc_folder + filename, 'r') as input:
            content = input.readlines()
            content = " ".join([line.strip() for line in content])

        for filter_type in ['DBpedia:Person', 'DBpedia:Organisation', 'DBpedia:Place']:
            params = {'text': content, 
                      'types': filter_type,
                      'confidence': "0.5" 
                     }
            url = "http://localhost:2222/rest/annotate"
            resp = requests.post(url, data=params)
            try:
                result = json.loads(resp.text)
                if 'Resources' in result:
                    URIs = set([str(element['@URI']) for element in result['Resources']])
                    # print URIs
                    for URI in URIs:
                        if URI in dbpedia_freebase_links_dict:
                            output.write(dbpedia_freebase_links_dict[URI])
                            output.write(' ')
                            covered_count += 1
                        else:
                            print URI
                            missing_count += 1
                    if covered_count % 10 == 0:
                        print "Covered entities: " + str(covered_count) + "; Missing entities: " + str(missing_count)
            except:
                print resp.text
            output.write('\t')
        output.write('\n')


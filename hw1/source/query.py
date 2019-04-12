try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def process_query(query_path):
    root = ET.ElementTree(file=query_path).getroot()
    query_list = []
    query_id = []
    for topic in root.findall('topic'):
        query = []
        query_id.append(topic.find('number').text.strip()[-3:])
        concepts = topic.find('concepts').text.strip().strip('。').split('、')
        title = topic.find('title').text.strip()
        question = topic.find('question').text.strip().strip('。')
        question = question.replace('查詢', '#')
        question = question.replace('有關', '#')
        question = question.replace('以及', '#')
        question = question.replace('與', '#')
        #question = question.replace('之', '#')
        question = question.replace('，', '#')
        question = question.replace('、', '#')
        question = question.replace('。', '#')
        for voc in concepts + [question]:
            if len(voc) > 2: # slice into bigram
                #for c in voc: # unigram
                #    query.append(c)
                for i in range(len(voc) - 1):
                    query.append(voc[i:i+2])
            query.append(voc)
        query_list.append(query)
    return query_list, query_id
        


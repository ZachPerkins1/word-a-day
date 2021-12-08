from os.path import exists, getsize
import sqlite3, requests as _requests, itertools, re, os, json
import numpy as np
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import xml.etree.ElementTree as ET
from parts_of_speech_mapping import parts_of_speech_mapping

VERSION = '1'

requests = _requests.Session()

retries = Retry(total=5,
                backoff_factor=1,
                status_forcelist=[ 500, 502, 503, 504, 429 ])

requests.mount('http://', HTTPAdapter(max_retries=retries))
requests.mount('https://', HTTPAdapter(max_retries=retries))


wiktionary_file = 'enwiktionary-20211201-pages-meta-current.xml'
read_block_size_bytes = 1000000
get_frequency_batch_size = 20
db_insert_batch_size = 50


class DictionaryWord:
    def __init__(self, text: str, parts_of_speech: set):
        self.text = text
        self.parts_of_speech = parts_of_speech


class Word:
    def __init__(self, dictWord: DictionaryWord, popularity: float):
        self.text = dictWord.text
        self.popularity = popularity
        self.parts_of_speech = dictWord.parts_of_speech


class WiktionaryPage:
    def __init__(self, title: str, id: str, content: str):
        self.title = title
        self.id = id
        self.content = content

class WiktionaryContent:
    def __init__(self, contains_english: bool, parts_of_speech: set):
        self.contains_english = contains_english
        self.parts_of_speech = parts_of_speech

def generate(regen=False):
    db_file = f'datasets/v{VERSION}.sql'
    if (exists(db_file)):
        if regen:
            os.remove(db_file)
        else:
            raise Exception('Dataset for this version already exists')
        
    con = sqlite3.connect(db_file)
    cursor = con.cursor()
    create_tables(cursor)
    words_processed = 0

    for block in iterate_in_blocks(generate_words(), db_insert_batch_size):
        cursor.executemany(
            'INSERT INTO words (text, popularity, parts_of_speech) VALUES (?, ?, ?)', map(
                lambda word: [word.text, word.popularity, json.dumps(list(word.parts_of_speech))],
                block
            )
        )
        con.commit()
        words_processed += db_insert_batch_size
        print(f'words processed: {words_processed}')
    

def iterate_in_blocks(iterable, block_size):
    iterator = iter(iterable)

    count = 0
    def inner_iteration():
        nonlocal count
        for i in range(block_size):
            yield next(iterator)
            count += 1

    while count % block_size == 0:
        yield inner_iteration()


def create_tables(cursor):
    cursor.execute(
        '''
        CREATE TABLE words (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            popularity REAL,
            parts_of_speech TEXT
        )
        '''
    )


def should_filter_word(word_obj: Word):
    min_popularity_threshold = 0

    return word_obj.popularity <= min_popularity_threshold
            

def generate_words():
    worditer = read_words_from_dictionary()
    def get_next_words():
        return list(itertools.islice(worditer, get_frequency_batch_size))

    dictWords = get_next_words()

    while len(dictWords) > 0:
        # get rid of newlines
        frequencies = get_word_frequencies(list(map(lambda dictWord: dictWord.text, dictWords)))
        # print(frequencies)

        for word_obj in map(lambda dictWord: Word(dictWord, frequencies[dictWord.text]), dictWords):
            if not should_filter_word(word_obj):
                yield word_obj

        dictWords = get_next_words()

            

def read_words_from_dictionary():
    for block in read_wiktionary_page_block():
        page = parse_wiktionary_page_block(block)
        if not should_filter_page(page):
            content = parse_wiktionary_content(page.content)
            if not should_filter_content(content):
                yield DictionaryWord(page.title, content.parts_of_speech)

 # Iterate through all <page>content</page> and return the string content of the block including the surrounding tags
# Ensures that we only keep what we need loaded (because the full file is fucking massive).
def read_wiktionary_page_block():
    in_page = False
    curr_text = ''

    file_size = getsize(wiktionary_file)
    read_size = 0

    with open(wiktionary_file, 'r', encoding='utf-8') as f:
        new_block = f.read(read_block_size_bytes)
        while (len(new_block) > 0):
            curr_text += new_block
            while True:
                if in_page:
                    end_idx = curr_text.find('</page>')
                    if (end_idx >= 0):
                        end_idx_including_tag = end_idx + 7
                        yield curr_text[start_idx:end_idx_including_tag]
                        curr_text = curr_text[end_idx_including_tag:]
                        in_page = False
                    else:
                        break
                else:
                    start_idx = curr_text.find('<page>')
                    if (start_idx >= 0):
                        in_page = True
                    else:
                        break
                
            read_size += read_block_size_bytes
            print(f'{(read_size/file_size)*100}%')
            new_block = f.read(read_block_size_bytes)


def parse_wiktionary_page_block(block: str):
    root = ET.fromstring(block)
    revision = root.find('revision')

    return WiktionaryPage(
        root.findtext('title'),
        root.findtext('id'),
        revision.findtext('text') if revision is not None else None
    )

# Characters valid in english words
# This conveniently also filters out all the "Thing: Stuff" meta pages, e.g. "Appendix: Animals"
whitelisted_page_regex = re.compile('[a-zA-Z -\']+')

def should_filter_page(page: WiktionaryPage):    
    if page.content is None:
        return True

    whitelist_match = whitelisted_page_regex.match(page.title)
    # need to ensure match encompasses entire string
    if (whitelist_match and whitelist_match.start() == 0 and whitelist_match.end() == len(page.title)):
        return False

    return True

heading_regex = re.compile('[^=]==([\w ]+)==[^=]')
subheading_regex = re.compile('===([\w ]+)===')

def parse_wiktionary_content(content: str):
    heading_matches = list(heading_regex.finditer(content))

    # Find the index of the english heading
    english_idx = find_index(heading_matches, lambda match: match[1].lower().strip() == 'english' or match[1].lower().strip() == 'translingual')

    if english_idx < 0:
        return WiktionaryContent(False, None)

    startpos = heading_matches[english_idx].end()
    endpos = heading_matches[english_idx + 1].start() if english_idx < len(heading_matches) - 1 else len(content)

    subheading_matches = subheading_regex.finditer(
        content,
        pos=startpos,
        # specify endpos if we have another heading after the english heading
        endpos=endpos
    )


    parts_of_speech = set()
    for subheading_match in subheading_matches:
        part_of_speech = subheading_match[1]
        if part_of_speech in parts_of_speech_mapping:
            parts_of_speech.add(parts_of_speech_mapping[part_of_speech])
    
    return WiktionaryContent(True, parts_of_speech)

valid_parts_of_speech = set(['noun', 'adjective', 'verb', 'adverb', 'determiner', 'interjection'])
def should_filter_content(content: WiktionaryContent):
    if content.parts_of_speech is None:
        return True

    # Ensure that the words parts of speech are a subset of our accepted parts of speech
    for part_of_speech in content.parts_of_speech:
        if part_of_speech not in valid_parts_of_speech:
            return True
    
    return False

def find_index(lst, condition):
    for idx, item in enumerate(lst):
        if condition(item):
            return idx
    
    return -1
    


def get_word_frequencies(words):
    resp = requests.get('https://books.google.com/ngrams/json', params={
        'content': ','.join(words),
        'year_start': '2009',
        'year_end': '2019',
        'corpus': 26,
        'smoothing': 3
    })

    output_map = {}
    if (resp.status_code != 200):
        print(f'Non successful status code from ngrams: ${resp.status_code}')
        print(resp)
        print(resp.text)
        print(resp.headers)

    ngram_data = resp.json()


    for entry in ngram_data:
        output_map[entry['ngram']] = np.array(entry['timeseries']).mean()

    for word in words:
        if not word in output_map:
            # anything that ngram doesn't recognize, we will just set to 0
            output_map[word] = 0
    
    return output_map


generate(True)
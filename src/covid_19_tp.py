
import json

def open_files(filepaths, read_func=json.load):
    return [
        read_func(open(f'{filepath}'))
        for filepath in filepaths
    ]

def author_name(author_dict, name_format=['first', 'middle', 'last']):
    if not author_dict.get('last', None):
        return None
    return ' '.join(
        [
            ' '.join(author_dict[attribute]).strip()
            if isinstance(author_dict[attribute], list) 
            else author_dict[attribute].strip()
            for attribute in name_format
        ]
    )
def author_affiliation(affiliation_dict):
    return ', '.join(
        [
            value
            for value in [affiliation_dict.get('institution', None)]+list(affiliation_dict.get('location', {None: None}).values())
            if value
        ]
    )
def authors_name(name_list_dict, name_format=['first', 'middle', 'last'], affiliation=False):
    return ', '.join(
        [
            author_name(author_dict)+' ('+author_affiliation(author_dict['affiliation'])+')'
            if affiliation and author_dict.get('affiliation')
            else author_name(author_dict)
            for author_dict in name_list_dict
            if author_dict.get('last', None)
        ]
    )
def body_text(body_text):
    content = [
        (section['section'], section['text']) 
        for section in body_text
    ]
    content_dict = {
        section[0]: ''
        for section in content
    }
    for section, text in content:
        content_dict[section] += text

    return ''.join(
        [
            ' '.join([section, text, '\n\n'])
            for section, text in content_dict.items()
        ]
    )
def format_bib(bibs):
    if isinstance(bibs, dict): bibs = list(bibs.values())
    
    return '; '.join(
        ', '.join(
            [
                authors_name(bib[key]) if key=='authors' else str(bib[key])
                for key in ['title', 'authors', 'venue', 'year']
            ]
        )
        for bib in bibs
    )
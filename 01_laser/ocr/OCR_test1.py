import fitz
import openpyxl as px
from openpyxl.styles import Alignment

item_list = []

file_name = 'test.pdf'

doc = fitz.open(file_name)

print(f'{doc=}')

for page in range(len(doc)):
    text_blocks = doc[page].getText('blocks')

    for text_block in text_blocks:
        if not text_block[4].isspace():
            item_list.append([page, text_block[4]])
            print(f'{text_block[4]=}')


"""
There are so many ways a student can learn that could includ pdfs, text file and websites. 
This module is responsible for uploading the files and storing them in the database. 
It also provides an interface for the user to upload the files and view the uploaded files.
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader

notes_path = Path(__file__).resolve().parent / "notes.txt"
data = TextLoader(str(notes_path))
docs = data.load()
print(docs)



"""
Most important thing is every document has two things: the metadata and the page content.
The metadata is the information about the document such as the source, 
the page number, the author, etc.
"""
# Final Project for CSCI 205

All source code that was used for this project is found in this repository. Each table has its own dedicated python file for insertion into the vector database. The client folder contains a Flask app that hosts a web interface to retrieve documents from the database given a query and embedding table you want to retrieve documents from.

## Prerequisites
1. PostgreSQL Database with pgVector
2. OpenAI Key
3. Python 3.12 upwards

## Notes
1. This project uses .env file. If you plan on running this, consult the source code on the necessary environmental variables.
2. Make sure to have the database set up. Tables are automatically created by the python files, but source code needs to connect to a PostgreSQL Database with pgVector enabled.
3. Install the requirements.txt file first.

Please contact Gab for data used and db.sqlite files.
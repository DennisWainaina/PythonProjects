# This is a tutorial on how to connect to postgres using python
import psycopg2

# These are the details required to log-in into the database
DB_HOST = "localhost"
DB_NAME = "Library"
DB_USER = "postgres"
DB_PASS = '55895678'
NAME = 'Da Vinci Demons'

# This is the object used to connect to the database
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

# Used to execute queries
cur = conn.cursor()

# The queries
cur.execute("INSERT INTO books(book_id, author, type, name) VALUES(4, 'Dan Brown', 'Fiction', 'Digital Fortress');")
cur.execute('DELETE FROM books WHERE book_id = 4;')
cur.execute("INSERT INTO books(book_id, author, type, name) VALUES(4, 'Dan Brown', 'Fiction', 'Digital Fortress');")
cur.execute("SELECT * FROM books;")

# Used to save queries to execute
conn.commit()

# Used to show results of the query to use the select query must come last
print(cur.fetchall())

# Closing the cursor connection
cur.close()

# When you open a connection you have to close it
conn.close()

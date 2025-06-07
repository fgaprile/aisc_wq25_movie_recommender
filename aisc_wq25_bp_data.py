import psycopg2
from psycopg2.extensions import AsIs
from sqlalchemy import create_engine, text as sql_text

import requests
import lxml.html as lx
import re
import time
import nltk
import numpy as np

from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env


db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")


def db_create():
    """
    Creates a SQLAlchemy engine to connect to a PostgreSQL database,
    drops the existing 'movie_info' schema if it exists, then recreates
    the schema and a 'movies' table to store movie metadata.
    """
    db_eng = create_engine(
        f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}',
        connect_args={'options': '-c search_path=public'},
        isolation_level='READ COMMITTED'
    )
    print("Successfully created db engine.")

    # SQL commands to drop schema, create schema, and create table
    sql0 = "DROP SCHEMA IF EXISTS movie_info CASCADE;"
    sql1 = "CREATE SCHEMA IF NOT EXISTS movie_info;"
    sql2 = """
        CREATE TABLE IF NOT EXISTS movie_info.movies (
            title VARCHAR(100) PRIMARY KEY,
            actors TEXT[],       -- top 5 actors
            genres TEXT[],
            writers TEXT[],      -- top 2 writers
            director TEXT[],
            rt_score FLOAT,
            date VARCHAR(20)
        );
    """

    # Drop existing schema
    with db_eng.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(sql_text(sql0))
            transaction.commit()
            print('Database successfully refreshed!')
        except Exception as e:
            transaction.rollback()
            print('Error occured:', e)

    # Create schema
    with db_eng.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(sql_text(sql1))
            transaction.commit()
            print('Schema Created!!')
        except Exception as e:
            transaction.rollback()
            print('Error occured:', e)

    # Create movies table inside schema
    with db_eng.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(sql_text(sql2))
            transaction.commit()
            print('Table Created!!')
        except Exception as e:
            transaction.rollback()
            print('Error occured:', e)


def retrieval():
    """
    Scrapes the IMSDb alphabetical movie index pages and collects
    all movie script URLs into a list for further processing.
    """
    base = 'https://imsdb.com/alphabetical/'
    alphabet = ["0"] + list(map(chr, range(65, 91)))  # 0-9 + A-Z
    movieList = []

    # Iterate over each alphabetical page and extract movie links
    for character in alphabet:
        url = base + character
        response = requests.get(url)

        html = lx.fromstring(response.text)
        # XPath to find movie links on the page
        moviePaths = html.xpath('//*[@id="mainbody"]/table[2]/tr/td[3]//a/@href')
        movieList.extend(moviePaths)

    return movieList


def fetch_info(link):
    """
    For a given movie script page URL path, scrape metadata:
    - title
    - writers
    - genres
    - release date
    - Rotten Tomatoes score
    - director(s)
    - top 5 actors

    Returns a dictionary with this info.
    """
    dct = {'title': None, "writers": None, 'genres': None, 'date': None}
    base = 'https://imsdb.com'
    url = base + link

    response = requests.get(url)
    html = lx.fromstring(response.text)

    basePath = '//*[@id="mainbody"]//table[@class="script-details"]'

    # Extract movie title from the page header
    title = html.xpath("//table[@class = 'script-details']//td[@align = 'center' and @colspan = '2']//h1//text()")[0].split(" Script")[0]

    # Extract Script Date or Movie Release Date if available, fallback to empty string
    datePath = html.xpath('//*[@id="mainbody"]//table[@class="script-details"]//b[contains(text(), "Script Date")]/following-sibling::text()')
    movieDatePath = html.xpath('//*[@id="mainbody"]//table[@class="script-details"]//b[contains(text(), "Movie Release Date")]/following-sibling::text()')

    if datePath:
        search = [re.search(r'\b\d{4}\b', item).group(0) for item in datePath if re.search(r'\b\d{4}\b', item)]
        year = int(search[0])
    elif movieDatePath:
        search = [re.search(r'\b\d{4}\b', item).group(0) for item in movieDatePath if re.search(r'\b\d{4}\b', item)]
        year = int(search[0])
    else:
        year = ""

    # Extract writers and genres from links on the page
    paths = html.xpath(basePath + '//a/@href')
    writers = [re.search(r'w=(.+)', item).group(1) for item in paths if re.search(r'/writer\.php\?w=', item)]

    try:
        genres = [re.search(r'/genre/(\w+)', item).group(1) for item in paths if re.search(r'/genre/', item)]
    except:
        # Fallback if genre extraction by regex fails
        genres = html.xpath(basePath + "//a[contains(@href, '/genre')]//text()")

    dct['title'] = title
    dct['writers'] = writers
    dct['genres'] = genres
    dct['date'] = year

    def getTomatometer(movieName):
        """
        Helper function to fetch Rotten Tomatoes critic score, director(s),
        and top 5 actors from Rotten Tomatoes website for the movie.
        """
        base = 'http://rottentomatoes.com/m/'

        # Handle special cases with movie title formatting
        if movieName.lower().endswith(", the"):
            movieName = "The " + movieName[:-5].strip()

        movie = re.sub(r'[^\w\s]', '', movieName).replace(" ", "_").lower()
        url = base + movie

        response = requests.get(url)
        if response.status_code != 200:
            # Try alternate URL if first attempt fails
            url = base + movie.lower().replace('the_', "").strip()
            response = requests.get(url)
            if response.status_code != 200:
                return None

        html = lx.fromstring(response.text)
        # Extract critic score as a decimal
        score = html.xpath('//rt-text[contains(@slot, "criticsScore")]/text()')
        try:
            score = int(score[0].strip("%")) / 100
        except:
            score = None

        # Extract cast and crew info: identify director(s) and top 5 actors
        peoplePath = html.xpath("//section[contains(@class, 'cast-and-crew')]//div[@slot='insetText']")
        actorList = []
        directorList = []
        for person in peoplePath:
            name = person.xpath(".//p[@class='name']/text()")[0]
            role = person.xpath(".//p[@class='role']/text()")[0]
            if role == 'Director':
                directorList.append(name)
            else:
                actorList.append(name)

        return score, directorList[:1], actorList[:5]

    try:
        tomato, director, actors = getTomatometer(title)
        dct['rt_score'] = tomato
        dct['director'] = director
        dct['actors'] = actors
    except:
        return

    return dct


def write(dataList):
    """
    Inserts a list of movie dictionaries into the PostgreSQL 'movie_info.movies' table.
    Handles text arrays for fields like actors and writers with proper escaping.
    """
    connection = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cursor = connection.cursor()

    for dct in dataList:
        columns = dct.keys()
        # Format values, converting Python lists into PostgreSQL array syntax
        values = tuple(
            dct[column] if not isinstance(dct[column], list) else
            '{' + ','.join(f'"{x.replace("\"", "\\\"")}"' for x in dct[column]) + '}'  # Escape internal quotes
            for column in columns
        )
        tableInsert = f'INSERT INTO movie_info.movies ({", ".join(columns)}) VALUES ({", ".join(["%s"] * len(values))})'

        cursor.execute(tableInsert, values)
    connection.commit()
    cursor.close()
    connection.close()


# Create database schema and table
db_create()

# Retrieve all movie script links from IMSDb
linkList = retrieval()
linkList = list(set(linkList))  # Remove duplicates
print(len(linkList), "links")

# Fetch movie info for each link
infoList = []
count = 0
for link in linkList:
    info = fetch_info(link)
    if info:
        infoList.append(info)
    count += 1
    if (count % 10) == 0:
        print(count)

# Write all collected movie data to the database
print('writing data')
write(infoList)
print('success!')


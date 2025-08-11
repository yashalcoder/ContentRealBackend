import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))
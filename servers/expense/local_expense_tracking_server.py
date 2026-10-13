from fastmcp import FastMCP
import os
import json
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()
# Use temporary directory which should be writable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORIES_PATH = os.path.join(PROJECT_DIR, "categories.json")
DATABASE_URL = os.getenv("DATABASE_URL")

mcp = FastMCP("ExpenseTracker")

def init_db():  # Keep as sync for initialization
    # try:
    #     # Use synchronous sqlite3 just for initialization
    #     import sqlite3
    #     with sqlite3.connect(DB_PATH) as c:
    #         c.execute("PRAGMA journal_mode=WAL")
    #         c.execute("""
    #             CREATE TABLE IF NOT EXISTS expenses(
    #                 id INTEGER PRIMARY KEY AUTOINCREMENT,
    #                 date TEXT NOT NULL,
    #                 amount REAL NOT NULL,
    #                 category TEXT NOT NULL,
    #                 subcategory TEXT DEFAULT '',
    #                 note TEXT DEFAULT ''
    #             )
    #         """)
    #         print("Database initialized successfully")
    # except Exception as e:
    #     print(f"Database initialization error: {e}")
    #     raise
    with psycopg.connect(DATABASE_URL, autocommit=True) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS expenses(
                id SERIAL PRIMARY KEY,
                date TEXT NOT NULL,
                amount DOUBLE PRECISION NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT ''
            )
        """)

# Initialize database synchronously at module load
init_db()

async def get_db_conn():
    return await psycopg.AsyncConnection.connect(
        DATABASE_URL, autocommit=True, row_factory=dict_row
    )

@mcp.tool()
async def add_expense(date, amount, category, subcategory="", note=""):  # Changed: added async
    '''Add a new expense entry to the database.'''
    try:
        async with await get_db_conn() as c:  # Changed: added async
            cur = await c.execute(  # Changed: added await
                "INSERT INTO expenses(date, amount, category, subcategory, note)"
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (date, amount, category, subcategory, note)
            )
            row = await cur.fetchone()
            return {"status": "success", "id": row["id"], "message": "Expense added successfully"}
    except Exception as e:  # Changed: simplified exception handling
        return {"status": "error", "message": f"Database error: {str(e)}"}
    
@mcp.tool()
async def list_expenses(start_date, end_date):  # Changed: added async
    '''List expense entries within an inclusive date range.'''
    try:
        async with await get_db_conn() as c:  # Changed: added async
            cur = await c.execute(  # Changed: added await
                """
                SELECT id, date, amount, category, subcategory, note
                FROM expenses
                WHERE date BETWEEN %s AND %s
                ORDER BY date DESC, id DESC
                """,
                (start_date, end_date)
            )
            return await cur.fetchall()
    except Exception as e:
        return {"status": "error", "message": f"Error listing expenses: {str(e)}"}

@mcp.tool()
async def summarize(start_date, end_date, category=None):  # Changed: added async
    '''Summarize expenses by category within an inclusive date range.'''
    try:
        async with await get_db_conn() as c:  # Changed: added async
            query = """
                SELECT category, SUM(amount) AS total_amount, COUNT(*) as count
                FROM expenses
                WHERE date BETWEEN %s AND %s
            """
            params = [start_date, end_date]
            if category:
                query += " AND category = %s"
                params.append(category)
            query += " GROUP BY category ORDER BY total_amount DESC"

            cur = await c.execute(query, params)
            return await cur.fetchall()
    except Exception as e:
        return {"status": "error", "message": f"Error summarizing expenses: {str(e)}"}

@mcp.resource("expense:///categories", mime_type="application/json")  # Changed: expense:// → expense:///
def categories():
    try:    
        with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return json.dumps({"categories": [
            "Food & Dining", "Transportation", "Shopping", "Entertainment",
            "Bills & Utilities", "Healthcare", "Travel", "Education", "Business", "Other"
        ]}, indent=2)

# Start the server
if __name__ == "__main__":
    # mcp.run(transport="http", host="0.0.0.0", port=8000)
    mcp.run()
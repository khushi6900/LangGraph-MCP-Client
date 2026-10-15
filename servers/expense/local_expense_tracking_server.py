import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastmcp import FastMCP
import os
import json
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from datetime import date as _d, datetime, timedelta
from typing import Literal, Optional

load_dotenv()
# Use temporary directory which should be writable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORIES_PATH = os.path.join(PROJECT_DIR, "categories.json")
DATABASE_URL = os.getenv("DATABASE_URL")

# Load expense categories at startup for easy access
try:
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        cat_data = json.load(f)
        if isinstance(cat_data, dict):
            # Use keys as top‑level categories
            CATEGORIES_LIST = list(cat_data.keys())
        else:
            CATEGORIES_LIST = cat_data
except Exception:
    CATEGORIES_LIST = []

CATS = tuple(CATEGORIES_LIST) or ("Food & Dining","Transportation","Shopping","Entertainment",
    "Bills & Utilities","Healthcare","Travel","Education","Business","Other")

Category = Literal[CATS]

def _iso(s: Optional[str], default: _d) -> str:
    if not s:
        return default.isoformat()
    try:
        return datetime.strptime(s, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise ValueError(f"Invalid date format: {s}. Expected YYYY-MM-DD.")

mcp = FastMCP("ExpenseTracker")

def init_db():  # Keep as sync for initialization
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
async def add_expense(amount: float, category: Category, date: str, subcategory: str = "", note: str = ""):
    '''Add a new expense entry to the database.'''
    try:
        # Normalize category: if numeric or index, map to name if possible
        date = _iso(date, _d.today())
        if isinstance(category, (int, float)):
            idx = int(category)
            if 0 <= idx < len(CATEGORIES_LIST):
                category = CATEGORIES_LIST[idx]
            else:
                category = str(category)
        elif isinstance(category, str) and category.isdigit():
            idx = int(category)
            if 0 <= idx < len(CATEGORIES_LIST):
                category = CATEGORIES_LIST[idx]
            else:
                category = category

        # Normalize date: if integer or year‑only string, convert to ISO format (YYYY-01-01)
        if isinstance(date, int):
            date = f"{date:04d}-01-01"
        elif isinstance(date, str):
            if len(date) == 4 and date.isdigit():
                date = f"{date}-01-01"
            else:
                pass

        async with await get_db_conn() as c:
            cur = await c.execute(
                """
                INSERT INTO expenses(date, amount, category, subcategory, note)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
                """,
                (date, amount, category, subcategory, note)
            )
            row = await cur.fetchone()
            return {"status": "success", "id": row["id"], "message": "Expense added successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Database error: {str(e)}"}

@mcp.tool()
async def list_expenses(start_date: Optional[str]=None, end_date: Optional[str]=None):
    '''List expense entries within an inclusive date range.'''
    try:
        t= _d.today()
        start_date = _iso(start_date, t.replace(day=1))
        end_date = _iso(end_date, t)
        async with await get_db_conn() as c:
            cur = await c.execute(
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
async def summarize(start_date: Optional[str]=None, end_date: Optional[str]=None, category: Optional[str] = None):
    '''Summarize expenses by category within an inclusive date range.'''
    try:
        async with await get_db_conn() as c:
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

@mcp.resource("expense:///categories", mime_type="application/json")
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
    mcp.run()

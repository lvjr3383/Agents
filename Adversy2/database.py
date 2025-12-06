# database.py
import sqlite3
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime

DB_PATH = Path("cases.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            name TEXT,
            mode TEXT,
            timestamp TEXT,
            results_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_case(name: str, mode: str, results: dict) -> str:
    init_db()
    case_id = str(uuid4())[:12]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO cases (id, name, mode, timestamp, results_json)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        case_id,
        name,
        mode,
        datetime.now().isoformat(),
        json.dumps(results, ensure_ascii=False)
    ))
    conn.commit()
    conn.close()
    return case_id

def load_case(case_id: str = None, name: str = None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if case_id:
        c.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
    elif name:
        c.execute("SELECT * FROM cases WHERE name LIKE ? ORDER BY timestamp DESC LIMIT 1", (f"%{name}%",))
    else:
        c.execute("SELECT * FROM cases ORDER BY timestamp DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    if not rows:
        return None
    row = rows[0]
    return {
        "id": row[0],
        "name": row[1],
        "mode": row[2],
        "timestamp": row[3],
        "results": json.loads(row[4])
    }

def list_recent_cases(limit: int = 5):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, timestamp FROM cases ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "date": r[2][:10]} for r in rows]

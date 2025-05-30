import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PatternLogger:
    """
    Logs detected patterns/features to JSON or SQLite.
    Each pattern: name, time window, confidence, example candles (timestamps).
    """
    def __init__(self, backend: str = 'json', path: Optional[str] = None):
        self.backend = backend
        if not path:
            path = 'logs/patterns.json' if backend == 'json' else 'logs/patterns.db'
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if backend == 'sqlite':
            self._init_sqlite()
        elif backend == 'json':
            if not os.path.exists(self.path):
                with open(self.path, 'w') as f:
                    json.dump([], f)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _init_sqlite(self):
        self.conn = sqlite3.connect(self.path)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            time_window TEXT,
            confidence REAL,
            example_candles TEXT
        )''')
        self.conn.commit()

    def log_pattern(self, name: str, time_window: str, confidence: float, example_candles: List[Dict[str, Any]]):
        pattern = {
            'name': name,
            'time_window': time_window,
            'confidence': confidence,
            'example_candles': example_candles
        }
        if self.backend == 'json':
            self._log_json(pattern)
        elif self.backend == 'sqlite':
            self._log_sqlite(pattern)
        logger.info(f"Pattern logged: {name} ({time_window}), confidence={confidence}")

    def _log_json(self, pattern: Dict[str, Any]):
        with open(self.path, 'r+') as f:
            data = json.load(f)
            data.append(pattern)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    def _log_sqlite(self, pattern: Dict[str, Any]):
        self.conn.execute(
            "INSERT INTO patterns (name, time_window, confidence, example_candles) VALUES (?, ?, ?, ?)",
            (pattern['name'], pattern['time_window'], pattern['confidence'], json.dumps(pattern['example_candles']))
        )
        self.conn.commit()

    def close(self):
        if self.backend == 'sqlite':
            self.conn.close()

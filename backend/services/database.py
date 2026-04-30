"""
Database Service — SQLite storage for audit history.

Maintains a history of all accessibility audits performed,
allowing for historical tracking, statistics, and PDF generation
of past reports.
"""
import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, db_path: Optional[str] = None):
        from backend.config import settings
        self.db_path = Path(db_path or settings.DATABASE_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _get_connection(self):
        return sqlite3.connect(str(self.db_path))
        
    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT NOT NULL,
                        overall_score REAL NOT NULL,
                        grade TEXT NOT NULL,
                        total_issues INTEGER NOT NULL,
                        critical_count INTEGER NOT NULL,
                        warning_count INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        report_json TEXT NOT NULL,
                        ai_insights_json TEXT DEFAULT NULL
                    )
                ''')
                # Migration: add column if missing on older DBs
                try:
                    cursor.execute("ALTER TABLE audit_history ADD COLUMN ai_insights_json TEXT DEFAULT NULL")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            
    def save_audit(self, report_dict: Dict[str, Any]) -> int:
        """Save a new audit report to the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO audit_history 
                    (url, overall_score, grade, total_issues, critical_count, warning_count, timestamp, report_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    report_dict.get("url", ""),
                    report_dict.get("overall_score", 0.0),
                    report_dict.get("grade", "F"),
                    report_dict.get("total_issues", 0),
                    report_dict.get("critical_count", 0),
                    report_dict.get("warning_count", 0),
                    report_dict.get("timestamp", ""),
                    json.dumps(report_dict)
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to save audit: {e}")
            return -1
            
    def get_history(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve recent audit history (without full JSON to save bandwidth)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, url, overall_score, grade, total_issues, critical_count, warning_count, timestamp
                    FROM audit_history
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                ''', (limit, offset))
                
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []
            
    def get_audit_by_id(self, audit_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific full audit report by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT report_json FROM audit_history WHERE id = ?', (audit_id,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
        except Exception as e:
            logger.error(f"Failed to get audit {audit_id}: {e}")
            return None
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics for dashboard."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total audits
                cursor.execute('SELECT COUNT(*) FROM audit_history')
                total_audits = cursor.fetchone()[0]
                
                # Average score
                cursor.execute('SELECT AVG(overall_score) FROM audit_history')
                avg_score = cursor.fetchone()[0] or 0.0
                
                # Grade distribution
                cursor.execute('SELECT grade, COUNT(*) FROM audit_history GROUP BY grade')
                grade_dist = dict(cursor.fetchall())
                
                return {
                    "total_audits": total_audits,
                    "average_score": round(avg_score, 1),
                    "grade_distribution": grade_dist
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "total_audits": 0,
                "average_score": 0.0,
                "grade_distribution": {}
            }

    def save_ai_insights(self, audit_id: int, insights: Dict[str, Any]) -> bool:
        """Save AI insights for a specific audit."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE audit_history SET ai_insights_json = ? WHERE id = ?',
                    (json.dumps(insights), audit_id)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to save AI insights for audit {audit_id}: {e}")
            return False

    def get_ai_insights(self, audit_id: int) -> Optional[Dict[str, Any]]:
        """Get AI insights for a specific audit."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT ai_insights_json FROM audit_history WHERE id = ?', (audit_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
                return None
        except Exception as e:
            logger.error(f"Failed to get AI insights for audit {audit_id}: {e}")
            return None

db_service = DatabaseService()

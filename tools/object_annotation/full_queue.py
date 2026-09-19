"""Transactional stream queue. Completed output and unresolved review are separate."""
import json
import os
from pathlib import Path
import sqlite3
import time


def connect(root):
    db = sqlite3.connect(Path(root) / 'queue.sqlite', timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA busy_timeout=60000')
    return db


def initialize(root, jobs):
    with connect(root) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('''CREATE TABLE jobs (
            id INTEGER PRIMARY KEY, episode TEXT NOT NULL, camera TEXT NOT NULL,
            task INTEGER NOT NULL, frames INTEGER NOT NULL, priority INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            pid INTEGER, gpu TEXT, started REAL, updated REAL, processed INTEGER DEFAULT 0,
            seconds REAL, result TEXT, error TEXT, UNIQUE(episode,camera))''')
        db.executemany('INSERT INTO jobs(episode,camera,task,frames,priority) VALUES(?,?,?,?,?)', jobs)
        db.execute('CREATE INDEX job_status ON jobs(status,priority,id)')


def recover_dead(root):
    with connect(root) as db:
        for job in db.execute("SELECT id,pid FROM jobs WHERE status='running'").fetchall():
            try:
                os.kill(job['pid'], 0)
            except ProcessLookupError:
                db.execute("UPDATE jobs SET status='pending',pid=NULL,error='Worker exited; resumable retry' WHERE id=?",
                           (job['id'],))


def claim(root, gpu):
    with connect(root) as db:
        db.execute('BEGIN IMMEDIATE')
        job = db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY priority,id LIMIT 1").fetchone()
        if job is None:
            return None
        now = time.time()
        db.execute("UPDATE jobs SET status='running',attempts=attempts+1,pid=?,gpu=?,started=?,updated=?,processed=0 WHERE id=?",
                   (os.getpid(), str(gpu), now, now, job['id']))
        return dict(job) | {'attempts': job['attempts'] + 1}


def heartbeat(root, job, frames):
    with connect(root) as db:
        changed = db.execute("UPDATE jobs SET updated=?,processed=? WHERE id=? AND pid=? AND status='running'",
                             (time.time(), frames, job['id'], os.getpid())).rowcount
        if changed != 1:
            raise RuntimeError('Queue lease lost')


def finish(root, job, result):
    with connect(root) as db:
        changed = db.execute("UPDATE jobs SET status='done',processed=frames,updated=?,seconds=?,result=?,error=NULL WHERE id=? AND pid=? AND status='running'",
                             (time.time(), result['wall_seconds'], json.dumps(result), job['id'], os.getpid())).rowcount
        if changed != 1:
            raise RuntimeError('Queue lease lost at completion')


def fail(root, job, error):
    with connect(root) as db:
        db.execute("UPDATE jobs SET status=?,updated=?,error=? WHERE id=? AND pid=?",
                   ('pending' if job['attempts'] < 3 else 'failed', time.time(), error, job['id'], os.getpid()))


def status(root):
    with connect(root) as db:
        groups = [dict(r) for r in db.execute('SELECT status,count(*) AS streams,sum(frames) AS frames,sum(processed) AS processed FROM jobs GROUP BY status')]
        running = [dict(r) for r in db.execute("SELECT episode,camera,task,gpu,pid,processed,frames,started,updated FROM jobs WHERE status='running'")]
        recent = [json.loads(r['result']) for r in db.execute("SELECT result FROM jobs WHERE status='done' ORDER BY updated DESC LIMIT 100")]
        tasks = [dict(r) for r in db.execute("SELECT task,count(*) AS streams,sum(status='done') AS done,sum(status='failed') AS failed FROM jobs GROUP BY task")]
        completed_eps = db.execute("SELECT count(*) FROM (SELECT episode FROM jobs GROUP BY episode HAVING min(status='done')=1)").fetchone()[0]
    return dict(groups=groups, running=running, completed_episodes=completed_eps, tasks=tasks,
                recent_frames=sum(r['frames'] for r in recent),
                recent_worker_seconds=sum(r['wall_seconds'] for r in recent),
                recent_streams=len(recent),
                recent_review_events=sum(r.get('review_events', 0) for r in recent))

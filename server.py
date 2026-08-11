import http.server
import socketserver
import sqlite3
import json
import os
import urllib.parse
import math
from pathlib import Path

PORT = 8080
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mouse_events_2026-08-07.sqlite3")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def calculate_episode_metrics(rows):
    if not rows:
        return {}
    
    total_events = len(rows)
    start_t = rows[0]['t_monotonic_ns']
    end_t = rows[-1]['t_monotonic_ns']
    duration_ms = (end_t - start_t) / 1e6 if total_events > 1 else 0
    
    total_dist = 0.0
    speeds = []
    dt_list = []
    teleport_count = 0
    click_count = 0
    scroll_count = 0
    move_count = 0
    
    prev = None
    for r in rows:
        if r['event_type'] == 'click' or r['event_type'] == 'button':
            click_count += 1
        elif r['event_type'] == 'scroll':
            scroll_count += 1
        elif r['event_type'] == 'move':
            move_count += 1
            
        if prev is not None:
            dt_ns = r['t_monotonic_ns'] - prev['t_monotonic_ns']
            dt_ms = dt_ns / 1e6
            if dt_ms > 0:
                dt_list.append(dt_ms)
                dx = r['x'] - prev['x']
                dy = r['y'] - prev['y']
                dist = math.hypot(dx, dy)
                total_dist += dist
                speed = (dist / (dt_ms / 1000.0))
                speeds.append(speed)
                if speed > 5000:
                    teleport_count += 1
        prev = r
        
    start_x, start_y = rows[0]['x'], rows[0]['y']
    end_x, end_y = rows[-1]['x'], rows[-1]['y']
    direct_dist = math.hypot(end_x - start_x, end_y - start_y)
    
    efficiency = (direct_dist / total_dist) if total_dist > 0 else 1.0
    avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0
    
    # Calculate quality grade
    if total_events < 5 or duration_ms < 50:
        quality_grade = "MICRO_IDLE"
    elif teleport_count > 0:
        quality_grade = "TELEPORTATION"
    elif avg_speed > 3500 or max_speed > 8000:
        quality_grade = "NOISY"
    else:
        quality_grade = "GOLD"
        
    return {
        "event_count": total_events,
        "move_count": move_count,
        "click_count": click_count,
        "scroll_count": scroll_count,
        "duration_ms": round(duration_ms, 2),
        "total_distance_px": round(total_dist, 2),
        "direct_distance_px": round(direct_dist, 2),
        "efficiency_ratio": round(efficiency, 3),
        "avg_speed_px_s": round(avg_speed, 2),
        "max_speed_px_s": round(max_speed, 2),
        "teleport_count": teleport_count,
        "quality_grade": quality_grade
    }

class VisualizerHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress routine GET logging for performance
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/stats":
            self.handle_stats()
        elif path == "/api/episodes":
            self.handle_episodes(query)
        elif path == "/api/episode":
            self.handle_episode_detail(query)
        elif path == "/api/export":
            self.handle_export(query)
        elif path == "/api/simulate":
            self.handle_simulate(query)
        else:
            # Serve static files from working directory
            super().do_GET()

    def send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def handle_stats(self):
        conn = get_db_connection()
        c = conn.cursor()

        sessions = c.execute("SELECT session_id, file_date, started_utc_ns, monitors_json, mouse_settings_json FROM sessions").fetchall()
        session_list = [dict(s) for s in sessions]

        total_events = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        event_types = c.execute("SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type").fetchall()
        event_type_dict = {r['event_type']: r['cnt'] for r in event_types}

        episodes = c.execute("""
            SELECT session_id, episode_index, COUNT(*) as cnt,
                   MIN(t_monotonic_ns) as start_t, MAX(t_monotonic_ns) as end_t
            FROM events
            GROUP BY session_id, episode_index
        """).fetchall()

        total_episodes = len(episodes)
        gold_count = 0
        micro_count = 0
        noisy_count = 0

        dt_samples = []
        c.execute("SELECT t_monotonic_ns FROM events ORDER BY id LIMIT 5000")
        rows = c.fetchall()
        for i in range(1, len(rows)):
            dt_ms = (rows[i]['t_monotonic_ns'] - rows[i-1]['t_monotonic_ns']) / 1e6
            if 0 < dt_ms < 100:
                dt_samples.append(dt_ms)

        for ep in episodes:
            cnt = ep['cnt']
            dur_ms = (ep['end_t'] - ep['start_t']) / 1e6
            if cnt < 5 or dur_ms < 50:
                micro_count += 1
            else:
                gold_count += 1

        conn.close()

        readiness_score = round((gold_count / total_episodes) * 100, 1) if total_episodes > 0 else 0

        self.send_json({
            "total_events": total_events,
            "total_episodes": total_episodes,
            "gold_episodes": gold_count,
            "micro_episodes": micro_count,
            "readiness_score": readiness_score,
            "sessions": session_list,
            "event_types": event_type_dict,
            "sample_dt_ms": dt_samples[:500]
        })

    def handle_episodes(self, query):
        page = int(query.get('page', ['1'])[0])
        limit = int(query.get('limit', ['25'])[0])
        session_id = query.get('session_id', [None])[0]
        quality_filter = query.get('quality', ['ALL'])[0]
        min_events = int(query.get('min_events', ['0'])[0])
        has_click = query.get('has_click', ['false'])[0].lower() == 'true'

        offset = (page - 1) * limit
        conn = get_db_connection()
        c = conn.cursor()

        where_clauses = []
        params = []

        if session_id:
            where_clauses.append("session_id = ?")
            params.append(session_id)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        query_sql = f"""
            SELECT session_id, episode_index, COUNT(*) as event_count,
                   MIN(t_monotonic_ns) as start_t, MAX(t_monotonic_ns) as end_t,
                   SUM(CASE WHEN event_type = 'click' OR event_type = 'button' THEN 1 ELSE 0 END) as click_count,
                   SUM(CASE WHEN event_type = 'scroll' THEN 1 ELSE 0 END) as scroll_count
            FROM events
            {where_sql}
            GROUP BY session_id, episode_index
            HAVING event_count >= ?
            {"AND click_count > 0" if has_click else ""}
            ORDER BY episode_index ASC
        """
        params.append(min_events)

        all_episodes = c.execute(query_sql, params).fetchall()
        
        # Build enriched metrics & quality filter
        enriched = []
        for ep in all_episodes:
            dur_ms = (ep['end_t'] - ep['start_t']) / 1e6
            cnt = ep['event_count']
            if cnt < 5 or dur_ms < 50:
                q = "MICRO_IDLE"
            else:
                q = "GOLD"
                
            if quality_filter != "ALL" and q != quality_filter:
                continue

            enriched.append({
                "session_id": ep['session_id'],
                "episode_index": ep['episode_index'],
                "event_count": cnt,
                "duration_ms": round(dur_ms, 1),
                "click_count": ep['click_count'],
                "scroll_count": ep['scroll_count'],
                "quality_grade": q
            })

        total_matching = len(enriched)
        paged_items = enriched[offset:offset+limit]

        conn.close()

        self.send_json({
            "page": page,
            "limit": limit,
            "total_count": total_matching,
            "episodes": paged_items
        })

    def handle_episode_detail(self, query):
        session_id = query.get('session_id', [None])[0]
        episode_index = query.get('episode_index', [None])[0]

        if not session_id or episode_index is None:
            self.send_json({"error": "Missing session_id or episode_index"}, 400)
            return

        conn = get_db_connection()
        c = conn.cursor()

        rows = c.execute("""
            SELECT id, session_id, episode_index, event_index, t_monotonic_ns, t_utc_ns,
                   event_type, x, y, dx, dy, button, pressed, scroll_dx, scroll_dy, monitor_index
            FROM events
            WHERE session_id = ? AND episode_index = ?
            ORDER BY event_index ASC
        """, (session_id, int(episode_index))).fetchall()

        conn.close()

        event_list = [dict(r) for r in rows]
        metrics = calculate_episode_metrics(rows)

        # Calculate time series physics (v(t), a(t), dt)
        physics = []
        prev = None
        prev_speed = 0
        for i, r in enumerate(event_list):
            if i == 0:
                physics.append({"dt_ms": 0, "speed_px_s": 0, "accel_px_s2": 0, "dist_px": 0})
            else:
                dt_ns = r['t_monotonic_ns'] - prev['t_monotonic_ns']
                dt_ms = dt_ns / 1e6
                dx = r['x'] - prev['x']
                dy = r['y'] - prev['y']
                dist = math.hypot(dx, dy)
                speed = (dist / (dt_ms / 1000.0)) if dt_ms > 0 else 0
                accel = ((speed - prev_speed) / (dt_ms / 1000.0)) if dt_ms > 0 else 0
                prev_speed = speed
                physics.append({
                    "dt_ms": round(dt_ms, 2),
                    "speed_px_s": round(speed, 2),
                    "accel_px_s2": round(accel, 2),
                    "dist_px": round(dist, 2)
                })
            prev = r

        self.send_json({
            "session_id": session_id,
            "episode_index": int(episode_index),
            "metrics": metrics,
            "events": event_list,
            "physics": physics
        })

    def handle_export(self, query):
        min_events = int(query.get('min_events', ['5'])[0])
        normalize = query.get('normalize', ['true'])[0].lower() == 'true'

        conn = get_db_connection()
        c = conn.cursor()

        episodes = c.execute("""
            SELECT session_id, episode_index
            FROM events
            GROUP BY session_id, episode_index
            HAVING COUNT(*) >= ?
        """, (min_events,)).fetchall()

        dataset = []
        for ep in episodes:
            rows = c.execute("""
                SELECT t_monotonic_ns, event_type, x, y, dx, dy, button, pressed
                FROM events
                WHERE session_id = ? AND episode_index = ?
                ORDER BY event_index ASC
            """, (ep['session_id'], ep['episode_index'])).fetchall()

            if not rows:
                continue

            w, h = 1920.0, 1080.0
            trajectory = []
            t0 = rows[0]['t_monotonic_ns']

            for r in rows:
                dt_ms = (r['t_monotonic_ns'] - t0) / 1e6
                pt = {
                    "t_ms": round(dt_ms, 2),
                    "x": round(r['x'] / w, 5) if normalize else r['x'],
                    "y": round(r['y'] / h, 5) if normalize else r['y'],
                    "dx": r['dx'],
                    "dy": r['dy'],
                    "type": r['event_type'],
                    "button": r['button']
                }
                trajectory.append(pt)

            dataset.append({
                "episode_index": ep['episode_index'],
                "session_id": ep['session_id'],
                "length": len(trajectory),
                "trajectory": trajectory
            })

        conn.close()

        self.send_json({
            "total_exported": len(dataset),
            "normalized": normalize,
            "data": dataset
        })

    def handle_simulate(self, query):
        try:
            start_x = float(query.get('start_x', ['100'])[0])
            start_y = float(query.get('start_y', ['100'])[0])
            target_x = float(query.get('target_x', ['1200'])[0])
            target_y = float(query.get('target_y', ['700'])[0])

            import subprocess
            cmd = [
                r"C:\Users\moham\AppData\Local\Programs\Python\Python314\python.exe",
                "-c",
                f"from simulate_mouse import HumanMouseSimulator; import json; sim = HumanMouseSimulator(); print(json.dumps(sim.generate_trajectory(({start_x}, {start_y}), ({target_x}, {target_y}))))"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            if res.returncode == 0:
                traj = json.loads(res.stdout.strip())
                self.send_json({
                    "status": "success",
                    "start_pos": [start_x, start_y],
                    "target_pos": [target_x, target_y],
                    "point_count": len(traj),
                    "trajectory": traj
                })
            else:
                self.send_json({"error": res.stderr}, 500)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

def run_server():
    server_address = ('', PORT)
    httpd = socketserver.TCPServer(server_address, VisualizerHandler)
    print(f"Mouse Event Visualizer Server listening on http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped gracefully.")

if __name__ == '__main__':
    run_server()

from flask import Flask, request, jsonify, render_template, session, redirect, send_from_directory
import os, json, base64, hashlib, random, time
from datetime import datetime, date
from functools import wraps
import smtplib
from email.mime.text import MIMEText

# Lazy-load cv2 and numpy only when face recognition is needed
_cv2 = None
_np = None
def get_cv2():
    global _cv2, _np
    if _cv2 is None:
        import cv2 as __cv2
        import numpy as __np
        _cv2 = __cv2
        _np = __np
    return _cv2, _np

app = Flask(__name__)
app.secret_key = "faceroll_piyush_secret_key_2025"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

KNOWN_FACES_DIR = 'known_faces'
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs('uploads', exist_ok=True)

DATABASE_URL = os.environ.get('DATABASE_URL', '')
USE_PG = bool(DATABASE_URL)

if not USE_PG:
    import sqlite3
    os.makedirs('database', exist_ok=True)
    DB_PATH = 'database/attendance.db'

reset_codes = {}

# ── DB helpers ────────────────────────────────────────────────────────────────
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def get_db():
    if USE_PG:
        import psycopg2
        import psycopg2.extras
        url = DATABASE_URL
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
        conn = psycopg2.connect(url)
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def Q(sql):
    """Convert SQLite ? placeholders to %s for PostgreSQL."""
    return sql.replace('?', '%s') if USE_PG else sql

def fetchone(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(Q(sql), params)
    row = cur.fetchone()
    if row is None: return None
    if USE_PG:
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    return row

def fetchall(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(Q(sql), params)
    rows = cur.fetchall()
    if USE_PG:
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in rows]
    return rows

def execute(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(Q(sql), params)
    return cur

def executemany(conn, sql, rows):
    cur = conn.cursor()
    cur.executemany(Q(sql), rows)
    return cur

def lastrowid(conn, cur):
    if USE_PG:
        cur2 = conn.cursor()
        cur2.execute('SELECT lastval()')
        return cur2.fetchone()[0]
    return cur.lastrowid

# ── Init DB ───────────────────────────────────────────────────────────────────
def init_db():
    conn = get_db()
    if USE_PG:
        stmts = [
            """CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','teacher','student','parent')),
                class_name TEXT DEFAULT '',
                student_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS students (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                roll_number TEXT UNIQUE NOT NULL,
                class_name TEXT,
                face_images TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS attendance_sessions (
                id SERIAL PRIMARY KEY,
                session_name TEXT,
                class_name TEXT,
                teacher_id INTEGER,
                date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS attendance_records (
                id SERIAL PRIMARY KEY,
                session_id INTEGER,
                student_id INTEGER,
                status TEXT DEFAULT 'absent',
                confidence REAL,
                FOREIGN KEY(session_id) REFERENCES attendance_sessions(id),
                FOREIGN KEY(student_id) REFERENCES students(id)
            )""",
            """CREATE TABLE IF NOT EXISTS announcements (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                author_id INTEGER,
                target TEXT DEFAULT 'all',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS timetable (
                id SERIAL PRIMARY KEY,
                class_name TEXT NOT NULL,
                day TEXT NOT NULL,
                period INTEGER NOT NULL,
                subject TEXT NOT NULL,
                teacher TEXT NOT NULL,
                time_start TEXT DEFAULT '',
                time_end TEXT DEFAULT ''
            )""",
        ]
        for s in stmts:
            execute(conn, s)
        conn.commit()
    else:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','teacher','student','parent')),
                class_name TEXT DEFAULT '', student_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                roll_number TEXT UNIQUE NOT NULL, class_name TEXT,
                face_images TEXT DEFAULT '[]', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS attendance_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_name TEXT,
                class_name TEXT, teacher_id INTEGER, date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
                student_id INTEGER, status TEXT DEFAULT 'absent', confidence REAL,
                FOREIGN KEY(session_id) REFERENCES attendance_sessions(id),
                FOREIGN KEY(student_id) REFERENCES students(id));
            CREATE TABLE IF NOT EXISTS announcements (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
                body TEXT NOT NULL, author_id INTEGER, target TEXT DEFAULT 'all',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS timetable (
                id INTEGER PRIMARY KEY AUTOINCREMENT, class_name TEXT NOT NULL,
                day TEXT NOT NULL, period INTEGER NOT NULL, subject TEXT NOT NULL,
                teacher TEXT NOT NULL, time_start TEXT DEFAULT '', time_end TEXT DEFAULT '');
        ''')
        conn.commit()

    # Seed admin
    if not fetchone(conn, "SELECT id FROM users WHERE email='admin@school.com'"):
        execute(conn, "INSERT INTO users (name,email,password,role) VALUES (?,?,?,?)",
                ('Administrator','admin@school.com',hash_pw('admin123'),'admin'))
        conn.commit()
        admin = fetchone(conn, "SELECT id FROM users WHERE email='admin@school.com'")
        aid = admin['id']
        executemany(conn, "INSERT INTO announcements (title,body,author_id,target) VALUES (?,?,?,?)", [
            ('Welcome to FaceRoll!','Face-recognition attendance is now live.',aid,'all'),
            ('How to register faces','Go to Students → click "+ Faces" to upload photos.',aid,'all'),
        ])
        executemany(conn, "INSERT INTO timetable (class_name,day,period,subject,teacher,time_start,time_end) VALUES (?,?,?,?,?,?,?)", [
            ('10-A','Monday',1,'Mathematics','Mr. Ahmed','08:00','08:45'),
            ('10-A','Monday',2,'English','Ms. Priya','08:45','09:30'),
            ('10-A','Tuesday',1,'Chemistry','Ms. Sara','08:00','08:45'),
            ('10-B','Monday',1,'English','Ms. Priya','08:00','08:45'),
        ])
        conn.commit()
    conn.close()

init_db()

# ── Auth ──────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if 'user_id' not in session: return redirect('/login')
        return f(*a, **kw)
    return dec

def roles(*allowed):
    def deco(f):
        @wraps(f)
        def dec(*a, **kw):
            if session.get('role') not in allowed:
                return jsonify({'error':'Unauthorized'}), 403
            return f(*a, **kw)
        return dec
    return deco

# ── Face recognition (AI-powered via face_recognition library) ────────────────
_fr = None

def get_fr():
    global _fr
    if _fr is None:
        import face_recognition as __fr
        _fr = __fr
    return _fr

def detect_and_encode(image_path):
    """Load image and return face encodings."""
    fr = get_fr()
    import cv2 as cv
    img = cv.imread(image_path)
    if img is None: return []
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    locations = fr.face_locations(rgb, model='hog')
    encodings = fr.face_encodings(rgb, locations)
    return encodings

def encode_from_bytes(file_bytes):
    """Detect and encode faces from uploaded image bytes."""
    fr = get_fr()
    import numpy as _np
    import cv2 as cv
    arr = _np.frombuffer(file_bytes, _np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None: return [], []
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    locations = fr.face_locations(rgb, model='hog')
    encodings = fr.face_encodings(rgb, locations)
    return locations, encodings

def build_face_db(student_ids=None):
    """Build database of known face encodings."""
    conn = get_db()
    if student_ids:
        ph = ','.join(['%s' if USE_PG else '?']*len(student_ids))
        rows = fetchall(conn, f'SELECT id,name,face_images FROM students WHERE id IN ({ph})', student_ids)
    else:
        rows = fetchall(conn, 'SELECT id,name,face_images FROM students')
    conn.close()
    db = []
    for s in rows:
        for p in json.loads(s['face_images']):
            if not os.path.exists(p): continue
            encs = detect_and_encode(p)
            for enc in encs:
                db.append((s['id'], s['name'], enc))
    return db

def recognize_faces_in_image(file_bytes, face_db, tolerance=0.5):
    """Recognize faces in classroom photo. Returns list of (sid, name, location)."""
    if not face_db: return [], []
    fr = get_fr()
    import numpy as _np
    import cv2 as cv

    arr = _np.frombuffer(file_bytes, _np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None: return img, []

    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    locations = fr.face_locations(rgb, model='hog')
    encodings = fr.face_encodings(rgb, locations)

    known_encs = [d[2] for d in face_db]
    known_ids  = [d[0] for d in face_db]
    known_names= [d[1] for d in face_db]

    results = []
    for enc, loc in zip(encodings, locations):
        matches = fr.compare_faces(known_encs, enc, tolerance=tolerance)
        distances = fr.face_distance(known_encs, enc)
        if True in matches:
            best_idx = int(_np.argmin(distances))
            if matches[best_idx]:
                confidence = round((1 - distances[best_idx]) * 100, 1)
                results.append({
                    'sid': known_ids[best_idx],
                    'name': known_names[best_idx],
                    'location': loc,
                    'confidence': confidence
                })
                # Draw green box
                top,right,bottom,left = loc
                cv.rectangle(img,(left,top),(right,bottom),(52,211,153),2)
                cv.putText(img,known_names[best_idx].split()[0],(left,top-8),
                           cv.FONT_HERSHEY_SIMPLEX,0.5,(52,211,153),2)
            else:
                top,right,bottom,left = loc
                cv.rectangle(img,(left,top),(right,bottom),(59,130,246),2)
        else:
            top,right,bottom,left = loc
            cv.rectangle(img,(left,top),(right,bottom),(59,130,246),2)
    return img, results

# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route('/')
def root(): return redirect('/dashboard' if 'user_id' in session else '/login')

@app.route('/login')
def login_page(): return render_template('login.html')

@app.route('/signup')
def signup_page(): return render_template('signup.html')

@app.route('/static/manifest.json')
def manifest(): return send_from_directory('static','manifest.json',mimetype='application/manifest+json')

@app.route('/static/sw.js')
def sw(): return send_from_directory('static','sw.js',mimetype='application/javascript')

@app.route('/forgot-password')
def forgot_page(): return render_template('forgot.html')

@app.route('/dashboard')
@login_required
def dashboard(): return render_template('dashboard.html')

# ── Auth API ──────────────────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def do_login():
    d=request.json
    conn=get_db()
    user=fetchone(conn,'SELECT * FROM users WHERE email=? AND password=?',
                  (d.get('email',''), hash_pw(d.get('password',''))))
    conn.close()
    if not user: return jsonify({'error':'Invalid email or password'}),401
    session['user_id']=user['id']; session['name']=user['name']
    session['role']=user['role']; session['class']=user['class_name']
    session['student_id']=user['student_id']
    return jsonify({'success':True,'role':user['role']})

@app.route('/api/register', methods=['POST'])
def do_register():
    d=request.json
    if not all([d.get('name'),d.get('email'),d.get('password'),d.get('role')]):
        return jsonify({'error':'Name, email, password and role are required'}),400
    if d['role'] not in ('teacher','student','parent'):
        return jsonify({'error':'Invalid role'}),400
    if len(d['password'])<6:
        return jsonify({'error':'Password must be at least 6 characters'}),400
    try:
        conn=get_db()
        execute(conn,'INSERT INTO users (name,email,password,role,class_name) VALUES (?,?,?,?,?)',
                (d['name'],d['email'],hash_pw(d['password']),d['role'],d.get('class_name','')))
        conn.commit(); conn.close()
        return jsonify({'success':True})
    except Exception:
        return jsonify({'error':'An account with that email already exists'}),409

@app.route('/api/logout')
def do_logout(): session.clear(); return jsonify({'success':True})

@app.route('/api/me')
@login_required
def me(): return jsonify({'id':session['user_id'],'name':session['name'],
    'role':session['role'],'class':session.get('class'),'student_id':session.get('student_id')})

# ── Users ─────────────────────────────────────────────────────────────────────
@app.route('/api/users', methods=['GET'])
@login_required
@roles('admin')
def get_users():
    conn=get_db()
    rows=fetchall(conn,'SELECT id,name,email,role,class_name,created_at FROM users ORDER BY role,name')
    conn.close(); return jsonify(rows)

@app.route('/api/users', methods=['POST'])
@login_required
@roles('admin')
def add_user():
    d=request.json
    if not all([d.get('name'),d.get('email'),d.get('password'),d.get('role')]):
        return jsonify({'error':'name, email, password, role required'}),400
    try:
        conn=get_db()
        execute(conn,'INSERT INTO users (name,email,password,role,class_name,student_id) VALUES (?,?,?,?,?,?)',
                (d['name'],d['email'],hash_pw(d['password']),d['role'],d.get('class_name',''),d.get('student_id')))
        conn.commit(); conn.close(); return jsonify({'success':True})
    except Exception: return jsonify({'error':'Email already exists'}),409

@app.route('/api/users/<int:uid>', methods=['DELETE'])
@login_required
@roles('admin')
def del_user(uid):
    conn=get_db(); execute(conn,'DELETE FROM users WHERE id=?',(uid,)); conn.commit(); conn.close()
    return jsonify({'success':True})

# ── Students ──────────────────────────────────────────────────────────────────
@app.route('/api/students', methods=['GET'])
@login_required
def get_students():
    conn=get_db(); cls=request.args.get('class','')
    if session['role']=='student':
        rows=fetchall(conn,'SELECT * FROM students WHERE id=?',(session['student_id'],))
    elif cls:
        rows=fetchall(conn,'SELECT * FROM students WHERE class_name=? ORDER BY name',(cls,))
    else:
        rows=fetchall(conn,'SELECT * FROM students ORDER BY class_name,name')
    conn.close()
    return jsonify([{**r,'face_count':len(json.loads(r['face_images']))} for r in rows])

@app.route('/api/students', methods=['POST'])
@login_required
@roles('admin','teacher')
def add_student():
    d=request.json
    if not d.get('name') or not d.get('roll_number'):
        return jsonify({'error':'Name and roll number required'}),400
    try:
        conn=get_db()
        cur=execute(conn,'INSERT INTO students (name,roll_number,class_name) VALUES (?,?,?)',
                    (d['name'],d['roll_number'],d.get('class_name','')))
        conn.commit(); sid=lastrowid(conn,cur); conn.close()
        return jsonify({'success':True,'id':sid})
    except Exception: return jsonify({'error':'Roll number already exists'}),409

@app.route('/api/students/<int:sid>', methods=['DELETE'])
@login_required
@roles('admin')
def del_student(sid):
    conn=get_db()
    row=fetchone(conn,'SELECT face_images FROM students WHERE id=?',(sid,))
    if row:
        for p in json.loads(row['face_images']):
            if os.path.exists(p): os.remove(p)
        execute(conn,'DELETE FROM students WHERE id=?',(sid,))
        execute(conn,'DELETE FROM attendance_records WHERE student_id=?',(sid,))
        conn.commit()
    conn.close(); return jsonify({'success':True})

@app.route('/api/students/<int:sid>/face', methods=['POST'])
@login_required
@roles('admin','teacher')
def upload_face(sid):
    conn=get_db()
    s=fetchone(conn,'SELECT * FROM students WHERE id=?',(sid,))
    if not s: conn.close(); return jsonify({'error':'Not found'}),404
    existing=json.loads(s['face_images']); saved=0
    import cv2 as _cv2, numpy as _np
    for f in request.files.getlist('images'):
        file_bytes = f.read()
        arr = _np.frombuffer(file_bytes, _np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if img is None: continue
        # Check face exists using face_recognition
        locs, encs = encode_from_bytes(file_bytes)
        if not encs: continue
        ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
        fname = f'{KNOWN_FACES_DIR}/s{sid}_{len(existing)+saved}_{ts}.png'
        _cv2.imwrite(fname, img); existing.append(fname); saved+=1
    execute(conn,'UPDATE students SET face_images=? WHERE id=?',(json.dumps(existing),sid))
    conn.commit(); conn.close()
    if not saved: return jsonify({'error':'No clear face detected. Use a well-lit, front-facing photo.'}),400
    return jsonify({'success':True,'saved':saved,'total':len(existing)})

@app.route('/api/classes')
@login_required
def get_classes():
    conn=get_db()
    rows=fetchall(conn,"SELECT DISTINCT class_name FROM students WHERE class_name!='' ORDER BY class_name")
    conn.close(); return jsonify([r['class_name'] for r in rows])

# ── Attendance ────────────────────────────────────────────────────────────────
@app.route('/api/attendance/mark', methods=['POST'])
@login_required
@roles('admin','teacher')
def mark_attendance():
    sess_name=request.form.get('session_name',f'Session {datetime.now().strftime("%d %b %Y %H:%M")}')
    class_name=request.form.get('class_name','')
    file=request.files.get('photo')
    if not file: return jsonify({'error':'No photo uploaded'}),400
    file_bytes = file.read()
    conn=get_db()
    students=(fetchall(conn,'SELECT * FROM students WHERE class_name=?',(class_name,))
              if class_name else fetchall(conn,'SELECT * FROM students'))
    if not students: conn.close(); return jsonify({'error':'No students found. Add students first.'}),400
    if not sum(len(json.loads(s['face_images'])) for s in students):
        conn.close(); return jsonify({'error':'No face photos uploaded yet.'}),400
    sids=[s['id'] for s in students]
    face_db=build_face_db(sids)
    if not face_db: conn.close(); return jsonify({'error':'No face encodings found. Re-upload student photos.'}),400
    cur=execute(conn,'INSERT INTO attendance_sessions (session_name,class_name,teacher_id,date) VALUES (?,?,?,?)',
                (sess_name,class_name,session['user_id'],date.today().isoformat()))
    sess_id=lastrowid(conn,cur); conn.commit()
    for sid in sids:
        execute(conn,'INSERT INTO attendance_records (session_id,student_id,status) VALUES (?,?,?)',(sess_id,sid,'absent'))
    conn.commit()
    img_out, results = recognize_faces_in_image(file_bytes, face_db)
    marked=[]
    recognized=[]
    for r in results:
        sid=r['sid']
        if sid in sids and sid not in marked:
            execute(conn,'UPDATE attendance_records SET status=?,confidence=? WHERE session_id=? AND student_id=?',
                    ('present',float(r['confidence']),sess_id,sid))
            s=fetchone(conn,'SELECT name,roll_number FROM students WHERE id=?',(sid,))
            recognized.append({'name':s['name'],'roll':s['roll_number'],'confidence':r['confidence']})
            marked.append(sid)
    conn.commit()
    records=fetchall(conn,'''SELECT s.name,s.roll_number,ar.status,ar.confidence
        FROM attendance_records ar JOIN students s ON s.id=ar.student_id
        WHERE ar.session_id=? ORDER BY s.name''',(sess_id,))
    conn.close()
    att=[{'name':r['name'],'roll':r['roll_number'],'status':r['status'],'confidence':r['confidence']} for r in records]
    import cv2 as _cv
    _,buf=_cv.imencode('.jpg',img_out,[_cv.IMWRITE_JPEG_QUALITY,82])
    return jsonify({'success':True,'faces_detected':len(results),'recognized':recognized,
                    'attendance':att,'present_count':len(marked),'total_students':len(students),
                    'annotated_image':'data:image/jpeg;base64,'+base64.b64encode(buf).decode()})

@app.route('/api/attendance/history')
@login_required
def att_history():
    conn=get_db(); cls=request.args.get('class','')
    if session['role']=='student' and session.get('student_id'):
        rows=fetchall(conn,'''SELECT ar.status,ar.confidence,ass.session_name,ass.date,ass.class_name
            FROM attendance_records ar JOIN attendance_sessions ass ON ass.id=ar.session_id
            WHERE ar.student_id=? ORDER BY ass.created_at DESC''',(session['student_id'],))
        conn.close(); return jsonify(rows)
    q='SELECT * FROM attendance_sessions'+(f' WHERE class_name=?' if cls else '')+' ORDER BY created_at DESC LIMIT 50'
    sessions=fetchall(conn,q,(cls,) if cls else ())
    result=[]
    for s in sessions:
        counts=fetchall(conn,"SELECT status,COUNT(*) as c FROM attendance_records WHERE session_id=? GROUP BY status",(s['id'],))
        present=next((c['c'] for c in counts if c['status']=='present'),0)
        absent=next((c['c'] for c in counts if c['status']=='absent'),0)
        result.append({'id':s['id'],'session_name':s['session_name'],'class_name':s['class_name'],
                       'date':s['date'],'present':present,'absent':absent,'total':present+absent})
    conn.close(); return jsonify(result)

@app.route('/api/attendance/session/<int:sid>')
@login_required
def sess_detail(sid):
    conn=get_db()
    s=fetchone(conn,'SELECT * FROM attendance_sessions WHERE id=?',(sid,))
    if not s: conn.close(); return jsonify({'error':'Not found'}),404
    records=fetchall(conn,'''SELECT s.name,s.roll_number,ar.status,ar.confidence
        FROM attendance_records ar JOIN students s ON s.id=ar.student_id
        WHERE ar.session_id=? ORDER BY s.name''',(sid,))
    conn.close(); return jsonify({'session':s,'records':records})

# ── Announcements ─────────────────────────────────────────────────────────────
@app.route('/api/announcements', methods=['GET'])
@login_required
def get_announcements():
    conn=get_db()
    rows=fetchall(conn,'''SELECT a.*,u.name as author_name FROM announcements a
        LEFT JOIN users u ON u.id=a.author_id ORDER BY a.created_at DESC LIMIT 30''')
    conn.close(); return jsonify(rows)

@app.route('/api/announcements', methods=['POST'])
@login_required
@roles('admin','teacher')
def add_announcement():
    d=request.json
    if not d.get('title') or not d.get('body'): return jsonify({'error':'Title and body required'}),400
    conn=get_db()
    execute(conn,'INSERT INTO announcements (title,body,author_id,target) VALUES (?,?,?,?)',
            (d['title'],d['body'],session['user_id'],d.get('target','all')))
    conn.commit(); conn.close(); return jsonify({'success':True})

@app.route('/api/announcements/<int:aid>', methods=['DELETE'])
@login_required
@roles('admin')
def del_announcement(aid):
    conn=get_db(); execute(conn,'DELETE FROM announcements WHERE id=?',(aid,)); conn.commit(); conn.close()
    return jsonify({'success':True})

# ── Timetable ─────────────────────────────────────────────────────────────────
@app.route('/api/timetable', methods=['GET'])
@login_required
def get_timetable():
    cls=request.args.get('class',''); conn=get_db()
    rows=(fetchall(conn,'SELECT * FROM timetable WHERE class_name=? ORDER BY day,period',(cls,))
          if cls else fetchall(conn,'SELECT * FROM timetable ORDER BY class_name,day,period'))
    conn.close(); return jsonify(rows)

@app.route('/api/timetable', methods=['POST'])
@login_required
@roles('admin')
def add_timetable():
    d=request.json; conn=get_db()
    execute(conn,'INSERT INTO timetable (class_name,day,period,subject,teacher,time_start,time_end) VALUES (?,?,?,?,?,?,?)',
            (d['class_name'],d['day'],int(d['period']),d['subject'],d['teacher'],d.get('time_start',''),d.get('time_end','')))
    conn.commit(); conn.close(); return jsonify({'success':True})

@app.route('/api/timetable/<int:tid>', methods=['DELETE'])
@login_required
@roles('admin')
def del_timetable(tid):
    conn=get_db(); execute(conn,'DELETE FROM timetable WHERE id=?',(tid,)); conn.commit(); conn.close()
    return jsonify({'success':True})

# ── Stats ─────────────────────────────────────────────────────────────────────
@app.route('/api/stats')
@login_required
def stats():
    conn=get_db()
    d={
        'students':fetchone(conn,'SELECT COUNT(*) as c FROM students')['c'],
        'teachers':fetchone(conn,"SELECT COUNT(*) as c FROM users WHERE role='teacher'")['c'],
        'sessions':fetchone(conn,'SELECT COUNT(*) as c FROM attendance_sessions')['c'],
        'announcements':fetchone(conn,'SELECT COUNT(*) as c FROM announcements')['c'],
    }
    conn.close(); return jsonify(d)

# ── Profile ───────────────────────────────────────────────────────────────────
@app.route('/api/profile', methods=['POST'])
@login_required
def update_profile():
    d=request.json
    if not d.get('name') or not d.get('email'): return jsonify({'error':'Name and email required'}),400
    try:
        conn=get_db()
        execute(conn,'UPDATE users SET name=?,email=?,class_name=? WHERE id=?',
                (d['name'],d['email'],d.get('class_name',''),session['user_id']))
        conn.commit(); conn.close()
        session['name']=d['name']; session['class']=d.get('class_name','')
        return jsonify({'success':True})
    except Exception: return jsonify({'error':'Email already in use by another account'}),409

@app.route('/api/change-password', methods=['POST'])
@login_required
def change_password():
    d=request.json
    if not d.get('old_password') or not d.get('new_password'):
        return jsonify({'error':'Both old and new passwords required'}),400
    if len(d['new_password'])<6: return jsonify({'error':'New password must be at least 6 characters'}),400
    conn=get_db()
    user=fetchone(conn,'SELECT * FROM users WHERE id=? AND password=?',
                  (session['user_id'],hash_pw(d['old_password'])))
    if not user: conn.close(); return jsonify({'error':'Current password is incorrect'}),401
    execute(conn,'UPDATE users SET password=? WHERE id=?',(hash_pw(d['new_password']),session['user_id']))
    conn.commit(); conn.close(); return jsonify({'success':True})

# ── Password reset ────────────────────────────────────────────────────────────
@app.route('/api/send-reset-code', methods=['POST'])
def send_reset_code():
    d=request.json; email=d.get('email','').strip().lower()
    if not email: return jsonify({'error':'Email is required'}),400
    conn=get_db()
    user=fetchone(conn,'SELECT id,name FROM users WHERE LOWER(email)=?',(email,))
    conn.close()
    if not user: return jsonify({'error':'No account found with that email'}),404
    code=str(random.randint(100000,999999))
    reset_codes[email]={'code':code,'expires':time.time()+600}
    mail_user=os.environ.get('MAIL_EMAIL','')
    mail_pass=os.environ.get('MAIL_PASSWORD','')
    if not mail_user or not mail_pass: return jsonify({'error':'Email service not configured'}),500
    try:
        msg=MIMEText(f"Hello {user['name']},\n\nYour FaceRoll password reset code is:\n\n  {code}\n\nThis code expires in 10 minutes.\n\n— FaceRoll Portal")
        msg['Subject']=f'FaceRoll — Your Reset Code: {code}'
        msg['From']=mail_user; msg['To']=email
        with smtplib.SMTP_SSL('smtp.gmail.com',465) as s:
            s.login(mail_user,mail_pass); s.sendmail(mail_user,email,msg.as_string())
        return jsonify({'success':True})
    except Exception as e:
        return jsonify({'error':'Failed to send email. Check server config.'}),500

@app.route('/api/verify-reset-code', methods=['POST'])
def verify_reset_code():
    d=request.json; email=d.get('email','').strip().lower(); code=d.get('code','').strip()
    entry=reset_codes.get(email)
    if not entry: return jsonify({'error':'No code sent. Please request a new one.'}),400
    if time.time()>entry['expires']:
        del reset_codes[email]; return jsonify({'error':'Code expired. Please request a new one.'}),400
    if entry['code']!=code: return jsonify({'error':'Incorrect code. Please try again.'}),400
    entry['verified']=True; return jsonify({'success':True})

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    d=request.json; email=d.get('email','').strip().lower(); new_pw=d.get('new_password','')
    if not email or not new_pw: return jsonify({'error':'Email and new password required'}),400
    if len(new_pw)<6: return jsonify({'error':'Password must be at least 6 characters'}),400
    entry=reset_codes.get(email)
    if not entry or not entry.get('verified'): return jsonify({'error':'Please verify your code first'}),403
    conn=get_db()
    user=fetchone(conn,'SELECT id FROM users WHERE LOWER(email)=?',(email,))
    if not user: conn.close(); return jsonify({'error':'No account found'}),404
    execute(conn,'UPDATE users SET password=? WHERE LOWER(email)=?',(hash_pw(new_pw),email))
    conn.commit(); conn.close(); del reset_codes[email]
    return jsonify({'success':True})

if __name__ == '__main__':
    print('\n🎓 FaceRoll — DISTRICT CM SOE PORTAL')
    print('   http://localhost:5000')
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=True)

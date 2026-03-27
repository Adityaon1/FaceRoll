from flask import Flask, request, jsonify, render_template, session, redirect
import cv2, numpy as np, os, json, base64, sqlite3, hashlib, secrets
from datetime import datetime, date
from functools import wraps

app = Flask(__name__)
app.secret_key = "faceroll_piyush_secret_key_2025"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

DB_PATH      = 'database/attendance.db'
KNOWN_FACES_DIR = 'known_faces'
for d in [KNOWN_FACES_DIR, 'database', 'uploads']:
    os.makedirs(d, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','teacher','student','parent')),
            class_name TEXT DEFAULT '',
            student_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            class_name TEXT,
            face_images TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT,
            class_name TEXT,
            teacher_id INTEGER,
            date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            student_id INTEGER,
            status TEXT DEFAULT 'absent',
            confidence REAL,
            FOREIGN KEY(session_id) REFERENCES attendance_sessions(id),
            FOREIGN KEY(student_id) REFERENCES students(id)
        );
        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            body  TEXT NOT NULL,
            author_id INTEGER,
            target TEXT DEFAULT 'all',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS timetable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            day TEXT NOT NULL,
            period INTEGER NOT NULL,
            subject TEXT NOT NULL,
            teacher TEXT NOT NULL,
            time_start TEXT DEFAULT '',
            time_end   TEXT DEFAULT ''
        );
    ''')
    # Seed default admin
    if not conn.execute("SELECT id FROM users WHERE email='admin@school.com'").fetchone():
        conn.execute("INSERT INTO users (name,email,password,role) VALUES (?,?,?,?)",
                     ('Administrator','admin@school.com',hash_pw('admin123'),'admin'))
        # Sample announcements
        conn.executemany("INSERT INTO announcements (title,body,author_id,target) VALUES (?,?,?,?)",[
            ('Welcome to FaceRoll!','Face-recognition attendance is now live. Teachers can mark attendance by uploading a classroom photo.',1,'all'),
            ('How to register faces','Go to Students → click "+ Faces" next to a student and upload 2–5 clear front-facing photos.',1,'all'),
        ])
        # Sample timetable
        conn.executemany("INSERT INTO timetable (class_name,day,period,subject,teacher,time_start,time_end) VALUES (?,?,?,?,?,?,?)",[
            ('10-A','Monday',   1,'Mathematics','Mr. Ahmed',   '08:00','08:45'),
            ('10-A','Monday',   2,'English',    'Ms. Priya',   '08:45','09:30'),
            ('10-A','Monday',   3,'Physics',    'Mr. Rajan',   '09:45','10:30'),
            ('10-A','Tuesday',  1,'Chemistry',  'Ms. Sara',    '08:00','08:45'),
            ('10-A','Tuesday',  2,'Biology',    'Mr. Kumar',   '08:45','09:30'),
            ('10-A','Wednesday',1,'Mathematics','Mr. Ahmed',   '08:00','08:45'),
            ('10-A','Wednesday',2,'History',    'Ms. Noor',    '08:45','09:30'),
            ('10-B','Monday',   1,'English',    'Ms. Priya',   '08:00','08:45'),
            ('10-B','Monday',   2,'Mathematics','Mr. Ahmed',   '08:45','09:30'),
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

# ── Face recognition ──────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier("/data/data/com.termux/files/usr/share/opencv4/haarcascades/" + 'haarcascade_frontalface_default.xml')
FACE_SIZE = (100,100)

def detect_faces_in_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for scale,nbrs in [(1.1,6),(1.05,4),(1.03,3)]:
        faces = face_cascade.detectMultiScale(gray,scaleFactor=scale,minNeighbors=nbrs,minSize=(50,50))
        if len(faces): return gray, faces
    return gray, np.array([])

def preprocess_face(gray,x,y,w,h):
    return cv2.GaussianBlur(cv2.equalizeHist(cv2.resize(gray[y:y+h,x:x+w],FACE_SIZE)),(3,3),0)

def face_to_features(f):
    hists=[]
    for s in [1,2,4]:
        h,w=f.shape; sh,sw=max(1,h//s),max(1,w//s)
        for r in range(s):
            for c in range(s):
                cell=f[r*sh:(r+1)*sh,c*sw:(c+1)*sw]
                hist=cv2.normalize(cv2.calcHist([cell],[0],None,[32],[0,256]),None).flatten()
                hists.append(hist)
    return np.concatenate(hists)

def build_face_db(student_ids=None):
    conn=get_db()
    if student_ids:
        ph=','.join(['?']*len(student_ids))
        rows=conn.execute(f'SELECT id,name,face_images FROM students WHERE id IN ({ph})',student_ids).fetchall()
    else:
        rows=conn.execute('SELECT id,name,face_images FROM students').fetchall()
    conn.close()
    db=[]
    for s in rows:
        for p in json.loads(s['face_images']):
            if not os.path.exists(p): continue
            img=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            db.append((s['id'],s['name'],face_to_features(cv2.equalizeHist(cv2.resize(img,FACE_SIZE)))))
    return db

def recognize_face(fg,db,thr=0.78):
    if not db: return None,0.0
    feats=face_to_features(fg); scores={}
    for sid,_,df in db:
        d=np.dot(feats,df); n=np.linalg.norm(feats)*np.linalg.norm(df)
        scores.setdefault(sid,[]).append(float(d/n) if n else 0.0)
    avg={sid:float(np.mean(v)) for sid,v in scores.items()}
    best=max(avg,key=avg.get)
    return (best,avg[best]) if avg[best]>=thr else (None,avg[best])

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def root():
    return redirect('/dashboard' if 'user_id' in session else '/login')

@app.route('/login', methods=['GET'])
def login_page(): return render_template('login.html')

@app.route('/signup', methods=['GET'])
def signup_page(): return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard(): return render_template('dashboard.html')

# ══════════════════════════════════════════════════════════════════════════════
# AUTH API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/login', methods=['POST'])
def do_login():
    d=request.json
    conn=get_db()
    user=conn.execute('SELECT * FROM users WHERE email=? AND password=?',
                      (d.get('email',''),hash_pw(d.get('password','')))).fetchone()
    conn.close()
    if not user: return jsonify({'error':'Invalid email or password'}),401
    session['user_id']   = user['id']
    session['name']      = user['name']
    session['role']      = user['role']
    session['class']     = user['class_name']
    session['student_id']= user['student_id']
    return jsonify({'success':True,'role':user['role']})

@app.route('/api/register', methods=['POST'])
def do_register():
    d = request.json
    if not all([d.get('name'), d.get('email'), d.get('password'), d.get('role')]):
        return jsonify({'error': 'Name, email, password and role are required'}), 400
    if d['role'] not in ('admin', 'teacher', 'student', 'parent'):
        return jsonify({'error': 'Invalid role'}), 400
    if len(d['password']) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    try:
        conn = get_db()
        conn.execute(
            'INSERT INTO users (name, email, password, role, class_name) VALUES (?,?,?,?,?)',
            (d['name'], d['email'], hash_pw(d['password']), d['role'], d.get('class_name', ''))
        )
        conn.commit(); conn.close()
        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'An account with that email already exists'}), 409

@app.route('/api/logout')
def do_logout():
    session.clear(); return jsonify({'success':True})

@app.route('/api/me')
@login_required
def me():
    return jsonify({'id':session['user_id'],'name':session['name'],
                    'role':session['role'],'class':session.get('class'),
                    'student_id':session.get('student_id')})

# ══════════════════════════════════════════════════════════════════════════════
# USERS (admin)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/users', methods=['GET'])
@login_required
@roles('admin')
def get_users():
    conn=get_db()
    rows=conn.execute('SELECT id,name,email,role,class_name,created_at FROM users ORDER BY role,name').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/users', methods=['POST'])
@login_required
@roles('admin')
def add_user():
    d=request.json
    if not all([d.get('name'),d.get('email'),d.get('password'),d.get('role')]):
        return jsonify({'error':'name, email, password, role required'}),400
    try:
        conn=get_db()
        conn.execute('INSERT INTO users (name,email,password,role,class_name,student_id) VALUES (?,?,?,?,?,?)',
                     (d['name'],d['email'],hash_pw(d['password']),d['role'],
                      d.get('class_name',''),d.get('student_id')))
        conn.commit(); conn.close()
        return jsonify({'success':True})
    except sqlite3.IntegrityError:
        return jsonify({'error':'Email already exists'}),409

@app.route('/api/users/<int:uid>', methods=['DELETE'])
@login_required
@roles('admin')
def del_user(uid):
    conn=get_db(); conn.execute('DELETE FROM users WHERE id=?',(uid,)); conn.commit(); conn.close()
    return jsonify({'success':True})

# ══════════════════════════════════════════════════════════════════════════════
# STUDENTS
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/students', methods=['GET'])
@login_required
def get_students():
    conn=get_db()
    cls=request.args.get('class','')
    if session['role']=='student':
        rows=conn.execute('SELECT * FROM students WHERE id=?',(session['student_id'],)).fetchall()
    elif cls:
        rows=conn.execute('SELECT * FROM students WHERE class_name=? ORDER BY name',(cls,)).fetchall()
    else:
        rows=conn.execute('SELECT * FROM students ORDER BY class_name,name').fetchall()
    conn.close()
    return jsonify([{**dict(r),'face_count':len(json.loads(r['face_images']))} for r in rows])

@app.route('/api/students', methods=['POST'])
@login_required
@roles('admin','teacher')
def add_student():
    d=request.json
    if not d.get('name') or not d.get('roll_number'):
        return jsonify({'error':'Name and roll number required'}),400
    try:
        conn=get_db()
        conn.execute('INSERT INTO students (name,roll_number,class_name) VALUES (?,?,?)',
                     (d['name'],d['roll_number'],d.get('class_name','')))
        conn.commit()
        sid=conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.close()
        return jsonify({'success':True,'id':sid})
    except sqlite3.IntegrityError:
        return jsonify({'error':'Roll number already exists'}),409

@app.route('/api/students/<int:sid>', methods=['DELETE'])
@login_required
@roles('admin')
def del_student(sid):
    conn=get_db()
    row=conn.execute('SELECT face_images FROM students WHERE id=?',(sid,)).fetchone()
    if row:
        for p in json.loads(row['face_images']):
            if os.path.exists(p): os.remove(p)
        conn.execute('DELETE FROM students WHERE id=?',(sid,))
        conn.execute('DELETE FROM attendance_records WHERE student_id=?',(sid,))
        conn.commit()
    conn.close(); return jsonify({'success':True})

@app.route('/api/students/<int:sid>/face', methods=['POST'])
@login_required
@roles('admin','teacher')
def upload_face(sid):
    conn=get_db()
    s=conn.execute('SELECT * FROM students WHERE id=?',(sid,)).fetchone()
    if not s: conn.close(); return jsonify({'error':'Not found'}),404
    existing=json.loads(s['face_images']); saved=0
    for f in request.files.getlist('images'):
        arr=np.frombuffer(f.read(),np.uint8)
        img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        if img is None: continue
        gray,faces=detect_faces_in_img(img)
        if not len(faces): continue
        x,y,w,h=faces[0]
        fi=preprocess_face(gray,x,y,w,h)
        ts=datetime.now().strftime('%Y%m%d%H%M%S%f')
        fname=f'{KNOWN_FACES_DIR}/s{sid}_{len(existing)+saved}_{ts}.png'
        cv2.imwrite(fname,fi); existing.append(fname); saved+=1
    conn.execute('UPDATE students SET face_images=? WHERE id=?',(json.dumps(existing),sid))
    conn.commit(); conn.close()
    if not saved: return jsonify({'error':'No clear face detected. Use a well-lit, front-facing photo.'}),400
    return jsonify({'success':True,'saved':saved,'total':len(existing)})

@app.route('/api/classes', methods=['GET'])
@login_required
def get_classes():
    conn=get_db()
    rows=conn.execute("SELECT DISTINCT class_name FROM students WHERE class_name!='' ORDER BY class_name").fetchall()
    conn.close()
    return jsonify([r['class_name'] for r in rows])

# ══════════════════════════════════════════════════════════════════════════════
# ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/attendance/mark', methods=['POST'])
@login_required
@roles('admin','teacher')
def mark_attendance():
    sess_name  = request.form.get('session_name',f'Session {datetime.now().strftime("%d %b %Y %H:%M")}')
    class_name = request.form.get('class_name','')
    file       = request.files.get('photo')
    if not file: return jsonify({'error':'No photo uploaded'}),400
    arr=np.frombuffer(file.read(),np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None: return jsonify({'error':'Invalid image'}),400

    conn=get_db()
    students=(conn.execute('SELECT * FROM students WHERE class_name=?',(class_name,))
              if class_name else conn.execute('SELECT * FROM students')).fetchall()
    if not students: conn.close(); return jsonify({'error':'No students found. Add students first.'}),400
    if not sum(len(json.loads(s['face_images'])) for s in students):
        conn.close(); return jsonify({'error':'No face photos uploaded yet.'}),400

    sids=[s['id'] for s in students]
    face_db=build_face_db(sids)
    gray,faces_found=detect_faces_in_img(img)

    cur=conn.execute('INSERT INTO attendance_sessions (session_name,class_name,teacher_id,date) VALUES (?,?,?,?)',
                     (sess_name,class_name,session['user_id'],date.today().isoformat()))
    sess_id=cur.lastrowid; conn.commit()
    for sid in sids:
        conn.execute('INSERT INTO attendance_records (session_id,student_id,status) VALUES (?,?,?)',(sess_id,sid,'absent'))
    conn.commit()

    marked,unknown,recognized=[],0,[]
    img_out=img.copy()
    for(x,y,w,h) in faces_found:
        fg=preprocess_face(gray,x,y,w,h)
        sid,score=recognize_face(fg,face_db)
        if sid and sid in sids and sid not in marked:
            conn.execute('UPDATE attendance_records SET status=?,confidence=? WHERE session_id=? AND student_id=?',
                         ('present',float(score),sess_id,sid))
            s=conn.execute('SELECT name,roll_number FROM students WHERE id=?',(sid,)).fetchone()
            recognized.append({'name':s['name'],'roll':s['roll_number'],'confidence':round(score*100,1)})
            marked.append(sid)
            cv2.rectangle(img_out,(x,y),(x+w,y+h),(52,211,153),2)
            cv2.putText(img_out,s['name'].split()[0],(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(52,211,153),2)
        else:
            unknown+=1
            cv2.rectangle(img_out,(x,y),(x+w,y+h),(59,130,246),2)
    conn.commit()
    records=conn.execute('''SELECT s.name,s.roll_number,ar.status,ar.confidence
        FROM attendance_records ar JOIN students s ON s.id=ar.student_id
        WHERE ar.session_id=? ORDER BY s.name''',(sess_id,)).fetchall()
    conn.close()
    att=[{'name':r['name'],'roll':r['roll_number'],'status':r['status'],'confidence':r['confidence']} for r in records]
    _,buf=cv2.imencode('.jpg',img_out,[cv2.IMWRITE_JPEG_QUALITY,82])
    return jsonify({'success':True,'faces_detected':int(len(faces_found)),'recognized':recognized,
                    'attendance':att,'present_count':len(marked),'total_students':len(students),
                    'annotated_image':'data:image/jpeg;base64,'+base64.b64encode(buf).decode()})

@app.route('/api/attendance/history')
@login_required
def att_history():
    conn=get_db()
    cls=request.args.get('class','')
    if session['role']=='student' and session.get('student_id'):
        # personal attendance records for a student
        rows=conn.execute('''SELECT ar.status,ar.confidence,ass.session_name,ass.date,ass.class_name
            FROM attendance_records ar JOIN attendance_sessions ass ON ass.id=ar.session_id
            WHERE ar.student_id=? ORDER BY ass.created_at DESC''',(session['student_id'],)).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    q='SELECT * FROM attendance_sessions'+(f' WHERE class_name=?' if cls else '')+' ORDER BY created_at DESC LIMIT 50'
    sessions=conn.execute(q,(cls,) if cls else ()).fetchall()
    result=[]
    for s in sessions:
        counts=conn.execute("SELECT status,COUNT(*) c FROM attendance_records WHERE session_id=? GROUP BY status",(s['id'],)).fetchall()
        present=next((c['c'] for c in counts if c['status']=='present'),0)
        absent =next((c['c'] for c in counts if c['status']=='absent'),0)
        result.append({'id':s['id'],'session_name':s['session_name'],'class_name':s['class_name'],
                       'date':s['date'],'present':present,'absent':absent,'total':present+absent})
    conn.close(); return jsonify(result)

@app.route('/api/attendance/session/<int:sid>')
@login_required
def sess_detail(sid):
    conn=get_db()
    s=conn.execute('SELECT * FROM attendance_sessions WHERE id=?',(sid,)).fetchone()
    if not s: conn.close(); return jsonify({'error':'Not found'}),404
    records=conn.execute('''SELECT s.name,s.roll_number,ar.status,ar.confidence
        FROM attendance_records ar JOIN students s ON s.id=ar.student_id
        WHERE ar.session_id=? ORDER BY s.name''',(sid,)).fetchall()
    conn.close()
    return jsonify({'session':dict(s),'records':[dict(r) for r in records]})

# ══════════════════════════════════════════════════════════════════════════════
# ANNOUNCEMENTS
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/announcements', methods=['GET'])
@login_required
def get_announcements():
    conn=get_db()
    rows=conn.execute('''SELECT a.*,u.name author_name FROM announcements a
        LEFT JOIN users u ON u.id=a.author_id ORDER BY a.created_at DESC LIMIT 30''').fetchall()
    conn.close(); return jsonify([dict(r) for r in rows])

@app.route('/api/announcements', methods=['POST'])
@login_required
@roles('admin','teacher')
def add_announcement():
    d=request.json
    if not d.get('title') or not d.get('body'): return jsonify({'error':'Title and body required'}),400
    conn=get_db()
    conn.execute('INSERT INTO announcements (title,body,author_id,target) VALUES (?,?,?,?)',
                 (d['title'],d['body'],session['user_id'],d.get('target','all')))
    conn.commit(); conn.close(); return jsonify({'success':True})

@app.route('/api/announcements/<int:aid>', methods=['DELETE'])
@login_required
@roles('admin')
def del_announcement(aid):
    conn=get_db(); conn.execute('DELETE FROM announcements WHERE id=?',(aid,)); conn.commit(); conn.close()
    return jsonify({'success':True})

# ══════════════════════════════════════════════════════════════════════════════
# TIMETABLE
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/timetable', methods=['GET'])
@login_required
def get_timetable():
    cls=request.args.get('class','')
    conn=get_db()
    rows=(conn.execute('SELECT * FROM timetable WHERE class_name=? ORDER BY day,period',(cls,))
          if cls else conn.execute('SELECT * FROM timetable ORDER BY class_name,day,period')).fetchall()
    conn.close(); return jsonify([dict(r) for r in rows])

@app.route('/api/timetable', methods=['POST'])
@login_required
@roles('admin')
def add_timetable():
    d=request.json
    conn=get_db()
    conn.execute('INSERT INTO timetable (class_name,day,period,subject,teacher,time_start,time_end) VALUES (?,?,?,?,?,?,?)',
                 (d['class_name'],d['day'],int(d['period']),d['subject'],d['teacher'],
                  d.get('time_start',''),d.get('time_end','')))
    conn.commit(); conn.close(); return jsonify({'success':True})

@app.route('/api/timetable/<int:tid>', methods=['DELETE'])
@login_required
@roles('admin')
def del_timetable(tid):
    conn=get_db(); conn.execute('DELETE FROM timetable WHERE id=?',(tid,)); conn.commit(); conn.close()
    return jsonify({'success':True})

# ══════════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/stats')
@login_required
def stats():
    conn=get_db()
    d={
        'students':      conn.execute('SELECT COUNT(*) FROM students').fetchone()[0],
        'teachers':      conn.execute("SELECT COUNT(*) FROM users WHERE role='teacher'").fetchone()[0],
        'sessions':      conn.execute('SELECT COUNT(*) FROM attendance_sessions').fetchone()[0],
        'announcements': conn.execute('SELECT COUNT(*) FROM announcements').fetchone()[0],
    }
    conn.close(); return jsonify(d)

if __name__ == '__main__':
    print('\n🎓 FaceRoll — School Portal')
    print('   http://localhost:5000')
    print('   Admin login: admin@school.com / admin123\n')
    import os; port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=True)

@app.route('/forgot-password', methods=['GET'])
def forgot_page(): return render_template('forgot.html')

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    d = request.json
    email = d.get('email','').strip()
    new_pw = d.get('new_password','')
    if not email or not new_pw:
        return jsonify({'error': 'Email and new password required'}), 400
    if len(new_pw) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    conn = get_db()
    user = conn.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()
    if not user:
        conn.close(); return jsonify({'error': 'No account found with that email'}), 404
    conn.execute('UPDATE users SET password=? WHERE email=?', (hash_pw(new_pw), email))
    conn.commit(); conn.close()
    return jsonify({'success': True})


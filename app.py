import os
import io
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, Response, redirect, url_for, session
from flask_cors import CORS
from flask_session import Session
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from apscheduler.schedulers.background import BackgroundScheduler
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

# Import Engines (Ensure these files exist in your folder)
from statistics_engine import StatisticsEngine, SchemaValidator
from report_engine import ReportEngine
from cleaning_engine import CleaningEngine
from ai_engine import AIEngine
import user_manager

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IS_VERCEL = os.environ.get('VERCEL', False)

if IS_VERCEL:
    UPLOAD_FOLDER = '/tmp/temp_uploads'
    SESSION_FILE_DIR = '/tmp/flask_session'
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'statflow-production-secret')
else:
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_uploads')
    SESSION_FILE_DIR = os.path.join(BASE_DIR, 'flask_session')
    app.config['SECRET_KEY'] = 'dev-secret-key'

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = SESSION_FILE_DIR
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_PERMANENT'] = False 
app.config['SESSION_USE_SIGNER'] = True

for d in [UPLOAD_FOLDER, SESSION_FILE_DIR]:
    if not os.path.exists(d):
        try: os.makedirs(d)
        except OSError: pass

Session(app)
user_manager.init_db()

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id): return user_manager.get_user(user_id)

def cleanup_temp_files():
    if not os.path.exists(UPLOAD_FOLDER): return
    now = time.time()
    cutoff = now - 3600
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.getmtime(file_path) < cutoff:
            try: os.remove(file_path)
            except: pass

scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_temp_files, trigger="interval", minutes=60)
scheduler.start()

def get_df_from_session(filename):
    if 'files' in session and filename in session['files']:
        return pd.read_json(io.StringIO(session['files'][filename]))
    return None

def save_df_to_session(filename, df):
    session['files'][filename] = df.to_json()
    session.modified = True

# --- HELPER: INTEGRITY CHECK ---
def check_duplicates(df):
    """Finds repeated rows and columns."""
    # 1. Repeated Rows
    dup_rows = df[df.duplicated(keep=False)]
    dup_row_indices = dup_rows.index.tolist()
    
    # 2. Repeated Columns (by content)
    dup_cols = []
    # Transpose to check columns as rows, or iterate
    # A simple check is identical column names (handled by pandas usually by renaming)
    # But checking identical content:
    for i in range(len(df.columns)):
        col_1 = df.iloc[:, i]
        for j in range(i + 1, len(df.columns)):
            col_2 = df.iloc[:, j]
            if col_1.equals(col_2):
                dup_cols.append(f"{df.columns[i]} == {df.columns[j]}")
                
    return dup_row_indices, dup_cols

# --- ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = user_manager.verify_user(request.form.get('username'), request.form.get('password'))
        if user:
            login_user(user)
            session['files'] = {}
            session['metadata'] = {}
            session['workflow_log'] = []
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=current_user.username)

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'files' not in request.files: return jsonify({'error': 'No files'}), 400
    files = request.files.getlist('files')
    
    if len(files) > 5:
        return jsonify({'error': 'Max 5 files allowed'}), 400

    if 'files' not in session: session['files'] = {}
    if 'workflow_log' not in session: session['workflow_log'] = []
    
    summary = []
    for file in files:
        if file.filename == '': continue
        try:
            filename = secure_filename(file.filename)
            # Basic Folder Check (files usually have extensions, folders don't or are rejected by read)
            if '.' not in filename: continue 

            buffer = io.BytesIO(file.read())
            
            # --- NEW: SPSS & SAS SUPPORT ---
            if filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(buffer)
            elif filename.endswith('.csv'):
                df = pd.read_csv(buffer)
            elif filename.endswith('.sav'):
                df, meta = pd.read_spss(buffer, metadata=True) # Requires pyreadstat
            elif filename.endswith('.sas7bdat'):
                df = pd.read_sas(buffer) # Pandas built-in or pyreadstat
            else:
                continue # Skip unsupported
            
            session['files'][filename] = df.to_json()
            session['workflow_log'].append(f"{datetime.now().strftime('%H:%M:%S')}: Uploaded {filename}")
            
            missing_count = int(df.isnull().sum().sum())
            dup_rows, dup_cols = check_duplicates(df)
            
            summary.append({
                'filename': filename, 
                'row_count': len(df),
                'col_count': len(df.columns),
                'missing_count': missing_count,
                'duplicate_rows': len(dup_rows),
                'duplicate_cols': dup_cols,
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'headers': list(df.columns)
            })
        except Exception as e: print(f"Error loading {file.filename}: {e}")
    
    session.modified = True
    return jsonify({'status': 'success', 'file_details': summary, 'files_uploaded': len(summary)})

@app.route('/pivot-data', methods=['POST'])
@login_required
def pivot_data():
    """
    NEW: Pivot Table functionality.
    """
    data = request.get_json()
    filename = data.get('filename')
    index_col = data.get('index')
    values_col = data.get('values')
    agg_func = data.get('agg', 'mean') # mean, sum, count
    
    df = get_df_from_session(filename)
    if df is None: return jsonify({'error': 'File not found'}), 404
    
    try:
        pivot = pd.pivot_table(df, index=index_col, values=values_col, aggfunc=agg_func)
        # Convert to dict for JSON
        pivot_data = pivot.reset_index().to_dict(orient='records')
        return jsonify({'status': 'success', 'pivot_data': pivot_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply-schema', methods=['POST'])
@login_required
def apply_schema():
    data = request.get_json()
    filename = data.get('filename')
    mapping = data.get('mapping')
    
    df = get_df_from_session(filename)
    if df is None: return jsonify({'error': 'File not found'}), 404
    
    try:
        new_df = SchemaValidator.map_columns(df, mapping)
        save_df_to_session(filename, new_df)
        session['workflow_log'].append(f"{datetime.now().strftime('%H:%M:%S')}: Applied schema mapping to {filename}")
        
        return jsonify({
            'status': 'success', 
            'new_headers': list(new_df.columns),
            'numeric_columns': new_df.select_dtypes(include=[np.number]).columns.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clean-data', methods=['POST'])
@login_required
def clean_data():
    """
    UPDATED: Now includes 'apply_rules' logic.
    """
    data = request.get_json()
    filename = data.get('filename')
    impute_method = data.get('impute_method')
    remove_outliers = data.get('remove_outliers')
    apply_rules = data.get('apply_rules') # NEW boolean
    
    df = get_df_from_session(filename)
    if df is None: return jsonify({'error': 'File not found'}), 404
    
    logs = []
    try:
        # 1. Imputation
        if impute_method and impute_method != 'none':
            df, l = CleaningEngine.impute_data(df, method=impute_method)
            logs.extend(l)
            
        # 2. Outliers
        if remove_outliers:
            df, l = CleaningEngine.remove_outliers(df, method='iqr')
            logs.extend(l)

        # 3. Rule Validation (NEW)
        if apply_rules:
            # For POC/Demo, we hardcode sensible defaults or infer them
            # Real version would take these from UI config
            rules = {
                'range': [
                    {'col': 'Age', 'min': 0, 'max': 120}, # Valid age range
                    {'col': 'Income', 'min': 0, 'max': 10000000} # Positive income
                ],
                'mandatory': [] # Add specific columns if critical
            }
            df, l = CleaningEngine.apply_rules(df, rules)
            logs.extend(l)
            
        save_df_to_session(filename, df)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        for log in logs:
            session['workflow_log'].append(f"{timestamp}: {log}")
            
        return jsonify({
            'status': 'success',
            'new_row_count': len(df),
            'logs': logs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-file-details', methods=['POST'])
@login_required
def get_file_details():
    filename = request.get_json().get('filename')
    df = get_df_from_session(filename)
    if df is None: return jsonify({'error': 'File not found'}), 404
    return jsonify({
        'filename': filename, 'headers': list(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
    })

@app.route('/calculate-estimates', methods=['POST'])
@login_required
def calculate_estimates():
    data = request.get_json()
    filename = data.get('filename')
    weight_col = data.get('weight_column')
    target_cols = data.get('target_columns')
    pop_total = data.get('population_total')
    
    df = get_df_from_session(filename)
    if df is None: return jsonify({'error': 'File not found'}), 404
    
    try:
        pop_control = float(pop_total) if pop_total and str(pop_total).strip() else None
        results = [StatisticsEngine.calculate_weighted_stats(df, col, weight_col, pop_control) 
                   for col in target_cols if col in df.columns]
        results = [r for r in results if r]
        
        session['last_results'] = results
        session['metadata'] = {'filename': filename, 'weight_col': weight_col, 'target_cols': target_cols, 'population_control_used': pop_control}
        return jsonify({'status': 'success', 'estimates': results})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/generate-visuals', methods=['POST'])
@login_required
def generate_visuals():
    meta = session.get('metadata', {})
    if not meta: return jsonify({'error': 'No analysis'}), 400
    df = get_df_from_session(meta['filename'])
    
    dist = ReportEngine.create_distribution_chart(df, meta['target_cols'][0])
    box = ReportEngine.create_boxplot(df, meta['target_cols'])
    ai_sum = AIEngine.generate_executive_summary(meta['filename'], session.get('last_results'), meta)
    
    session['charts'] = {'distribution_chart': dist, 'boxplot_chart': box}
    session['ai_summary'] = ai_sum
    return jsonify({'status': 'success', 'distribution_chart': dist, 'boxplot_chart': box, 'ai_summary': ai_sum})

@app.route('/download-pdf', methods=['GET'])
@login_required
def download_pdf():
    """
    Vercel-Safe PDF Handler:
    Since server-side PDF generation requires heavy system libraries unavailable on Vercel,
    we serve the HTML report with a script that auto-triggers the browser's Print dialog.
    """
    meta = session.get('metadata', {})
    stats = session.get('last_results', [])
    charts = session.get('charts', {})
    ai_summary = session.get('ai_summary', '')
    workflow_log = session.get('workflow_log', [])
    filename = meta.get('filename', 'report')

    # Generate the standard HTML report
    html_content = ReportEngine.generate_html_report(filename, stats, charts, meta, ai_summary)

    # Add Workflow Log
    log_html = "<h2>3. WORKFLOW LOG</h2><ul style='font-size:12px; color:#475569;'>" + \
               "".join([f"<li>{l}</li>" for l in workflow_log]) + "</ul>"
    html_content = html_content.replace("</body>", f"{log_html}</body>")

    # Inject Auto-Print Script
    # This forces the browser to open the "Save as PDF" dialog immediately
    print_script = "<script>window.onload = function() { window.print(); }</script>"
    html_content = html_content.replace("</body>", f"{print_script}</body>")

    return Response(
        html_content,
        mimetype="text/html",
        # We remove 'attachment' so it opens in the browser to trigger the print dialog
        headers={"Content-disposition": f"inline; filename=Report_{filename}.html"}
    )

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
from flask import Flask, render_template, request, jsonify
import json
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store data in memory (in production, use a database)
data_samples = {}
current_file_id = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jsonl', 'json'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_file_id
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(filepath)
        
        # Parse JSONL file
        samples = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        sample = json.loads(line)
                        sample['id'] = f"sample_{line_num + 1}"
                        samples.append(sample)
            
            data_samples[file_id] = samples
            current_file_id = file_id
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'num_samples': len(samples)
            })
        except Exception as e:
            return jsonify({'error': f'Error parsing file: {str(e)}'}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/get_samples')
def get_samples():
    if current_file_id and current_file_id in data_samples:
        samples_list = []
        for idx, sample in enumerate(data_samples[current_file_id]):
            samples_list.append({
                'id': sample['id'],
                'index': idx,
                'preview': sample.get('applied_functions', 'No function'),
                'type': sample.get('applied_functions', 'unknown')
            })
        return jsonify(samples_list)
    return jsonify([])

@app.route('/get_sample/<int:index>')
def get_sample(index):
    if current_file_id and current_file_id in data_samples:
        samples = data_samples[current_file_id]
        if 0 <= index < len(samples):
            return jsonify(samples[index])
    return jsonify({'error': 'Sample not found'}), 404

if __name__ == '__main__':
    # Create templates directory and index.html
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Viewer Dashboard</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-latex.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f5f5f5;
            height: 100vh;
            overflow: hidden;
        }

        .message-text {
        font-family: monospace;
        white-space: pre-wrap;
        }

        .container {
            display: flex;
            height: 100vh;
        }
        
        .sidebar {
            width: 300px;
            background-color: #2c3e50;
            color: white;
            overflow-y: auto;
            flex-shrink: 0;
        }
        
        .sidebar-header {
            padding: 20px;
            background-color: #34495e;
            border-bottom: 1px solid #1a252f;
        }
        
        .upload-btn {
            width: 100%;
            padding: 10px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }
        
        .upload-btn:hover {
            background-color: #2980b9;
        }
        
        .upload-input {
            display: none;
        }
        
        .sample-list {
            padding: 10px;
        }
        
        .sample-item {
            padding: 15px;
            margin-bottom: 5px;
            background-color: #34495e;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .sample-item:hover {
            background-color: #4a5f7a;
        }
        
        .sample-item.active {
            background-color: #3498db;
        }
        
        .sample-item-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .sample-item-type {
            font-size: 12px;
            opacity: 0.8;
            background-color: rgba(255,255,255,0.1);
            padding: 2px 8px;
            border-radius: 3px;
            display: inline-block;
        }
        
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .content-header {
            background-color: white;
            padding: 20px;
            border-bottom: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
        }
        
        .message-role {
            width: 80px;
            flex-shrink: 0;
            font-weight: bold;
            padding: 10px;
            text-align: center;
            border-radius: 5px;
            margin-right: 15px;
        }
        
        .message-role.user {
            background-color: #3498db;
            color: white;
        }
        
        .message-role.assistant {
            background-color: #2ecc71;
            color: white;
        }
        
        .message-content {
            flex: 1;
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            line-height: 1.6;
        }
        
        .think-highlight {
            background-color: #e8e8e8;
            padding: 2px 4px;
            border-radius: 3px;
        }
        
        .partial-solution-highlight {
            background-color: #fff3cd;
            padding: 2px 4px;
            border-radius: 3px;
            border: 1px solid #ffeaa7;
        }
        
        .message-content pre {
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 0;
            border: 1px solid #e1e4e8;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .message-content code {
            background-color: transparent;
            padding: 0;
            border-radius: 0;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .metadata {
            background-color: #fff;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metadata-item {
            margin-bottom: 10px;
        }
        
        .metadata-label {
            font-weight: bold;
            color: #555;
        }
        
        .empty-state {
            text-align: center;
            padding: 50px;
            color: #999;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h2 style="margin-bottom: 15px;">Data Viewer</h2>
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    📤 Upload JSONL File
                </button>
                <input type="file" id="fileInput" class="upload-input" accept=".jsonl,.json">
            </div>
            <div class="sample-list" id="sampleList">
                <div class="empty-state">No samples loaded</div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="content-header">
                <h1 id="contentTitle">Select a sample to view</h1>
            </div>
            <div class="chat-container" id="chatContainer">
                <div class="empty-state">
                    <h2>Welcome to Data Viewer</h2>
                    <p>Upload a JSONL file to get started</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Configure MathJax
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }
        };
        
        let currentSamples = [];
        let currentIndex = -1;
        
        // File upload handler
        document.getElementById('fileInput').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert(`Successfully loaded ${result.num_samples} samples`);
                    loadSamples();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error uploading file: ' + error.message);
            }
        });
        
        async function loadSamples() {
            try {
                const response = await fetch('/get_samples');
                currentSamples = await response.json();
                
                const sampleList = document.getElementById('sampleList');
                if (currentSamples.length === 0) {
                    sampleList.innerHTML = '<div class="empty-state">No samples loaded</div>';
                    return;
                }
                
                sampleList.innerHTML = currentSamples.map((sample, index) => `
                    <div class="sample-item" onclick="loadSample(${index})" id="sample-${index}">
                        <div class="sample-item-title">${sample.id}</div>
                        <span class="sample-item-type">${sample.type}</span>
                    </div>
                `).join('');
                
                // Load first sample automatically
                if (currentSamples.length > 0) {
                    loadSample(0);
                }
            } catch (error) {
                console.error('Error loading samples:', error);
            }
        }
        
        async function loadSample(index) {
            if (currentIndex !== -1) {
                document.getElementById(`sample-${currentIndex}`).classList.remove('active');
            }
            
            currentIndex = index;
            document.getElementById(`sample-${index}`).classList.add('active');
            
            try {
                const response = await fetch(`/get_sample/${index}`);
                const sample = await response.json();
                
                displaySample(sample);
            } catch (error) {
                console.error('Error loading sample:', error);
            }
        }
        
        function highlightContent(content) {
            // First escape HTML to prevent any rendering
            let escaped = content
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            
            // Highlight <think>...</think> sections in grey
            escaped = escaped.replace(/&lt;think&gt;([\s\S]*?)&lt;\/think&gt;/g, 
                '&lt;think&gt;<span class="think-highlight">$1</span>&lt;/think&gt;');
            
            // Highlight <partial_solution>...</partial_solution> sections
            escaped = escaped.replace(/&lt;partial_solution&gt;([\s\S]*?)&lt;\/partial_solution&gt;/g, 
                '&lt;partial_solution&gt;<span class="partial-solution-highlight">$1</span>&lt;/partial_solution&gt;');
            
            return escaped;
        }
        
        function displaySample(sample) {
            const contentTitle = document.getElementById('contentTitle');
            const chatContainer = document.getElementById('chatContainer');
            
            contentTitle.textContent = sample.id || 'Sample View';
            
            let html = '';
            
            // Display metadata
            if (sample.applied_functions) {
                html += `
                    <div class="metadata">
                        <div class="metadata-item">
                            <span class="metadata-label">Applied Functions:</span> ${sample.applied_functions}
                        </div>
                    </div>
                `;
            }
            
            // Display messages
            if (sample.messages && Array.isArray(sample.messages)) {
                sample.messages.forEach(message => {
                    const role = message.role;
                    const content = message.content;
                    
                    // Process content with highlighting
                    const highlightedContent = highlightContent(content);
                    
                    html += `
                        <div class="message">
                            <div class="message-role ${role}">${role.charAt(0).toUpperCase() + role.slice(1)}</div>
                            <div class="message-content message-text">${highlightedContent}</div>
                        </div>
                    `;
                });
            }
            
            chatContainer.innerHTML = html;
            
        }
        
        
        // Initial load
        loadSamples();
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Starting Flask Data Viewer Dashboard...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
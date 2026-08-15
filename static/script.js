// DOM Elements
const taskInput = document.getElementById('taskInput');
const languageSelect = document.getElementById('languageSelect');
const iterationsInput = document.getElementById('iterationsInput');
const forgeBtn = document.getElementById('forgeBtn');
const statusBadge = document.getElementById('statusBadge');
const welcomeState = document.getElementById('welcomeState');
const codeContainer = document.getElementById('codeContainer');
const reportContainer = document.getElementById('reportContainer');
const codeOutput = document.getElementById('codeOutput').querySelector('code');
const reportOutput = document.getElementById('reportOutput');
const fileName = document.getElementById('fileName');
const iterationCount = document.getElementById('iterationCount');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');

let currentCode = '';
let currentFilename = 'output.py';

// Event Listeners
forgeBtn.addEventListener('click', generateCode);
copyBtn.addEventListener('click', copyCode);
downloadBtn.addEventListener('click', downloadCode);

// Generate Code
async function generateCode() {
    const task = taskInput.value.trim();
    
    if (!task) {
        alert('Please enter a task description');
        return;
    }
    
    // Reset UI
    setStatus('generating');
    forgeBtn.disabled = true;
    forgeBtn.innerHTML = `
        <div class="spinner"></div>
        <span>Forging...</span>
    `;
    
    welcomeState.classList.add('hidden');
    codeContainer.classList.remove('hidden');
    reportContainer.classList.add('hidden');
    
    codeOutput.textContent = '// Generating code...';
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                task: task,
                language: languageSelect.value,
                max_iterations: parseInt(iterationsInput.value)
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            setStatus('success');
            currentCode = data.code || '';
            currentFilename = data.filename || 'output.py';
            
            fileName.textContent = currentFilename;
            iterationCount.textContent = `Iteration ${data.iterations}/${iterationsInput.value}`;
            codeOutput.textContent = currentCode;
            
            if (data.report) {
                reportContainer.classList.remove('hidden');
                reportOutput.textContent = data.report;
            }
        } else {
            setStatus('error');
            codeOutput.textContent = `// Error: ${data.error || 'Generation failed'}`;
        }
        
    } catch (error) {
        setStatus('error');
        codeOutput.textContent = `// Error: ${error.message}`;
    } finally {
        forgeBtn.disabled = false;
        forgeBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke-width="2"/>
            </svg>
            <span>Forge Code</span>
        `;
    }
}

// Copy Code
function copyCode() {
    if (!currentCode) return;
    
    navigator.clipboard.writeText(currentCode).then(() => {
        const originalHTML = copyBtn.innerHTML;
        copyBtn.innerHTML = '<span style="color: #10b981;">✓ Copied</span>';
        setTimeout(() => {
            copyBtn.innerHTML = originalHTML;
        }, 2000);
    });
}

// Download Code
function downloadCode() {
    if (!currentCode) return;
    
    const blob = new Blob([currentCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = currentFilename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Set Status
function setStatus(status) {
    statusBadge.className = `status-badge ${status}`;
    const statusText = {
        'idle': 'Idle',
        'generating': 'Generating',
        'success': 'Success',
        'error': 'Error'
    };
    statusBadge.textContent = statusText[status] || 'Idle';
}

// Add spinner CSS dynamically
const style = document.createElement('style');
style.textContent = `
    .spinner {
        width: 16px;
        height: 16px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-top-color: white;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);

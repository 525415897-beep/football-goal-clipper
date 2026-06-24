// electron/main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn, execSync } = require('child_process');

let mainWindow;
let cachedPythonPath = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, '..', 'frontend', 'index.html'));
}

function getPythonPath() {
  // Return cached path if already resolved
  if (cachedPythonPath) {
    return cachedPythonPath;
  }

  const isPackaged = app.isPackaged;

  if (!isPackaged) {
    // Development: use the project venv
    cachedPythonPath = path.join(__dirname, '..', 'backend', 'venv', 'bin', 'python3');
    return cachedPythonPath;
  }

  // Packaged app: no venv bundled, find system Python
  // Try python3 from PATH first
  try {
    const pythonFromPath = execSync('which python3', { encoding: 'utf-8' }).trim();
    if (pythonFromPath) {
      cachedPythonPath = pythonFromPath;
      return cachedPythonPath;
    }
  } catch (_) {
    // which failed, try fallback
  }

  // Fallback: check /usr/bin/python3
  const fallbackPath = '/usr/bin/python3';
  try {
    execSync(`test -x "${fallbackPath}"`);
    cachedPythonPath = fallbackPath;
    return cachedPythonPath;
  } catch (_) {
    // fallback not available either
  }

  // No Python found at all
  dialog.showErrorBox(
    'Python Not Found',
    'Python 3 is required to run this application.\n\n' +
    'Please install Python 3 from https://www.python.org/downloads/ ' +
    'or via Homebrew: brew install python3\n\n' +
    'Make sure python3 is available on your system PATH.'
  );
  app.quit();
  return 'python3'; // unreachable, satisfies return type
}

function getBackendPath() {
  const isPackaged = app.isPackaged;
  if (isPackaged) {
    return path.join(process.resourcesPath, 'backend');
  }
  return path.join(__dirname, '..', 'backend');
}

function checkPythonDeps() {
  const requiredModules = ['cv2', 'numpy'];
  try {
    execSync(
      `${getPythonPath()} -c "import ${requiredModules.join(', ')}"`,
      { encoding: 'utf-8', timeout: 15000 }
    );
    return true;
  } catch (_) {
    dialog.showErrorBox(
      'Missing Python Dependencies',
      'The following Python packages are required:\n\n' +
      `  pip3 install opencv-python numpy\n\n` +
      'Please install them using the command above in your terminal.'
    );
    app.quit();
    return false;
  }
}

function runPythonScript(scriptName, inputData) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(getBackendPath(), scriptName);
    const python = spawn(getPythonPath(), [scriptPath]);

    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => { stdout += data.toString(); });
    python.stderr.on('data', (data) => { stderr += data.toString(); });

    python.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Python exited with code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (e) {
        reject(new Error(`Failed to parse JSON: ${stdout}`));
      }
    });

    python.on('error', reject);
    python.stdin.write(JSON.stringify(inputData));
    python.stdin.end();
  });
}

// IPC handlers
ipcMain.handle('detect-goals', async (_, data) => {
  const result = await runPythonScript('detector.py', {
    video_path: data.videoPath,
    roi_left: data.roiLeft,
    roi_right: data.roiRight,
    params: data.params || {},
  });
  return result;
});

ipcMain.handle('export-clips', async (_, data) => {
  const result = await runPythonScript('exporter.py', {
    video_path: data.videoPath,
    timestamps: data.timestamps,
    output_dir: data.outputDir,
    before: data.before || 10,
    after: data.after || 10,
  });
  return result;
});

ipcMain.handle('select-video', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Videos', extensions: ['mp4', 'mov', 'm4v', 'avi', 'mkv'] }],
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('select-output-dir', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });
  return result.canceled ? null : result.filePaths[0];
});

app.whenReady().then(() => {
  if (!checkPythonDeps()) return;
  createWindow();
});
app.on('window-all-closed', () => app.quit());

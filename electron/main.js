// electron/main.js
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;

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
  const isPackaged = app.isPackaged;
  if (isPackaged) {
    return path.join(process.resourcesPath, 'backend', 'venv', 'bin', 'python3');
  }
  return path.join(__dirname, '..', 'backend', 'venv', 'bin', 'python3');
}

function getBackendPath() {
  const isPackaged = app.isPackaged;
  if (isPackaged) {
    return path.join(process.resourcesPath, 'backend');
  }
  return path.join(__dirname, '..', 'backend');
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

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());

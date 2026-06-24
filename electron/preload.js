// electron/preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  selectVideo: () => ipcRenderer.invoke('select-video'),
  selectOutputDir: () => ipcRenderer.invoke('select-output-dir'),
  detectGoals: (data) => ipcRenderer.invoke('detect-goals', data),
  exportClips: (data) => ipcRenderer.invoke('export-clips', data),
});

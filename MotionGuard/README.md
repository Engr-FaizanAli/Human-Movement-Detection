# MotionGuard

Production-grade Windows desktop application for MOG2-based motion detection.
Supports Dahua NVR, Hikvision DVR/NVR, direct IP cameras (RTSP/ONVIF),
and local video files — with polygon exclusion zones and multi-source grid display.

---

## Requirements

- Windows 10 / 11
- Python 3.10+
- The `ti_motion_detect_v4.py` algorithm file must be present at:
  `Human-Movement-Detection/motion_scripts/ti_motion_detect_v4.py`

---

## Setup (Development)

```bat
cd Human-Movement-Detection\MotionGuard

:: Create virtual environment
python -m venv .venv
.venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt

:: Run the application
python app\main.py
```

Application data is stored in `%APPDATA%\MotionGuard\`

---

## Usage Guide

### 1. Add a Dahua NVR

1. Click **Add Recorder** in the toolbar
2. Select Brand: **Dahua NVR**
3. Enter: Friendly Name, IP, RTSP Port (default 554), Username, Password, Channel Count
4. Stream Preference: **Substream** (recommended for low CPU usage)
5. Template: **Dahua NVR** (auto-selected)
6. Click **Test Channel** to verify connectivity
7. Click **Save Recorder** — channels CH1..CHN appear in the device tree

### 2. Add a Hikvision DVR/NVR

1. Click **Add Recorder**
2. Select Brand: **Hikvision DVR/NVR**
3. Fill in IP, Port, credentials, channel count
4. Template: **Hikvision (Standard)** uses `/Streaming/Channels/{ch*100+stream_index}`
   - Main stream: channel 101, 201, 301...
   - Substream: channel 102, 202, 302...
5. **If your DVR uses a different URL format**, select **Hikvision (Alt / Custom)**
   and enter your own template, e.g.:
   ```
   rtsp://{user}:{password}@{ip}:{port}/h264/ch{ch}/sub/av_stream
   ```
   Available variables: `{user}`, `{password}`, `{ip}`, `{port}`, `{ch}`, `{stream}`
6. Test and save

### 3. Scan ONVIF Cameras

1. Click **Scan ONVIF** — a WS-Discovery broadcast finds devices on the LAN
2. Select a discovered device, enter credentials, click **Resolve RTSP Stream**
3. The RTSP URL is auto-filled; click **Save Camera**

### 4. Add a Camera Manually (RTSP)

1. Click **Add Camera** → **Manual RTSP** tab
2. Enter the full RTSP URL, optionally add username/password separately
3. Click **Test Connection** to verify
4. Enter a name and click **Save Camera**

### 5. Run Detection on 1–3 Sources

1. Sources appear in the left **Devices** panel
2. Double-click a channel/camera to activate it (or right-click → Start)
3. The center grid adjusts automatically:
   - 1 source: full screen
   - 2 sources: side by side
   - 3 sources: 2×2 grid (one empty cell)
4. Motion indicators appear in each cell header
5. Events are listed in the right **Motion Events** panel

### 6. Offline Mode (Local Video File)

1. Click **Offline Mode** → select an MP4/AVI/MKV file
2. The file appears in Devices and auto-starts
3. Use Play/Pause/Seek controls at the bottom of the video cell
4. All detection features (polygons, alerts) work the same as live RTSP
5. Offline sources count toward the 3-source limit

### 7. Edit Polygon Exclusion Zones

1. Right-click a source in the device tree → **Edit Zones**
   (or click **Edit Zones** button in the video cell)
2. The editor opens with the current frame as background
3. **Left-click** to add vertices; **right-click** to close the polygon
4. **Ctrl+Z** undoes the last vertex; **Delete** removes selected zone
5. Click **Save Zones** — detection in those regions is immediately suppressed
6. Zones are stored normalized (0–1) and survive resolution changes

### 8. Settings

Click **Settings** in the toolbar to adjust:

| Tab | Controls |
|---|---|
| Detection | Detection mode (MOG2 / Frame Difference), sensitivity, background memory |
| Pre-Processing | Low-light contrast enhancement, temporal smoothing |
| Noise Filtering | Morphology, minimum/maximum object size, shape quality |
| Object Tracking | Confirmation frames, minimum movement, tracking tolerance |
| Alerts & Snapshots | Sound alarm volume, snapshot saving, retention policy |
| Performance | Detection FPS, preview FPS, stream preference |
| Application | Auto-start, minimize to tray, log level |

### 9. Diagnostics

Click **Diagnostics** to see:
- Local IP and hostname
- Active sources and last error per source
- RTSP URL tester (enter any URL to test connectivity)
- Log export (zips log files with credentials stripped)

---

## Building the Executable

### Step 1: Generate logo.ico

```bat
python make_ico.py
```

### Step 2: Build

```bat
pyinstaller motionguard.spec --clean
```

Output: `dist\MotionGuard\MotionGuard.exe`

Run the exe directly — no Python installation needed on the target machine.

### Optional: Inno Setup Installer

Create an installer with [Inno Setup](https://jrsoftware.org/isinfo.php):

```iss
[Setup]
AppName=MotionGuard
AppVersion=1.0
DefaultDirName={autopf}\MotionGuard
DefaultGroupName=MotionGuard
OutputBaseFilename=MotionGuard_Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\MotionGuard\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\MotionGuard"; Filename: "{app}\MotionGuard.exe"
Name: "{commondesktop}\MotionGuard"; Filename: "{app}\MotionGuard.exe"

[Run]
Filename: "{app}\MotionGuard.exe"; Description: "Launch MotionGuard"; Flags: nowait postinstall skipifsilent
```

---

## Project Structure

```
MotionGuard/
├── app/
│   ├── main.py                  Entry point
│   ├── core/
│   │   ├── motion_engine.py     Wraps ti_motion_detect_v4 (zero rewrites to v4)
│   │   ├── mask_engine.py       Normalized zone coordinates → pixel mask
│   │   ├── source_worker.py     QThread: per-source capture + detection loop
│   │   ├── source_manager.py    Manages up to 3 active workers
│   │   └── alerts.py            Sound alarm + snapshot saving + event logging
│   ├── ui/
│   │   ├── main_window.py       QMainWindow shell
│   │   ├── source_grid_widget.py  Adaptive 1/2/3-cell grid
│   │   ├── source_view_widget.py  Single source cell with controls
│   │   ├── polygon_editor.py    Polygon exclusion zone editor dialog
│   │   ├── offline_player_controls.py  Playback controls for local files
│   │   └── dialogs/
│   │       ├── add_recorder_dialog.py
│   │       ├── add_camera_dialog.py
│   │       ├── settings_dialog.py
│   │       └── diagnostics_dialog.py
│   ├── storage/
│   │   ├── db.py                SQLite thread-local connections, WAL mode
│   │   ├── migrations.py        Schema versioning
│   │   └── repositories.py      CRUD functions for all tables
│   └── utils/
│       ├── paths.py             %APPDATA%\MotionGuard\* path resolution
│       ├── logging_setup.py     Rotating file + console logging
│       ├── rtsp_templates.py    Dahua/Hikvision/Custom URL builders
│       ├── rtsp_test.py         Quick RTSP connectivity tester
│       └── net_discovery.py     ONVIF WS-Discovery + profile resolution
├── assets/
│   ├── icons/logo.png           Application logo
│   └── sounds/alarm.wav         Default alarm (auto-generated if missing)
├── make_ico.py                  PNG → ICO converter (run before build)
├── motionguard.spec             PyInstaller build spec
└── requirements.txt
```

---

## Data Storage

All persistent data is stored in `%APPDATA%\MotionGuard\`:

| Path | Contents |
|---|---|
| `config.db` | SQLite: recorders, channels, cameras, zones, settings, events |
| `logs\motionguard.log` | Rotating application log (credentials never logged) |
| `snapshots\` | Motion event snapshots (JPEG, retention-controlled) |

---

## Algorithm Integration Notes

`motion_engine.py` imports directly from `ti_motion_detect_v4.py` without any modification.
The wrapper uses: `TIPreprocessor`, `MotionDetector`, `MaskPostprocessor`, `ExclusionZoneManager`, `MotionVisualizer`.

To update the algorithm:
1. Drop in a newer `ti_motion_detect_v4.py`
2. Verify the class signatures are compatible with `EngineConfig`
3. No other code changes needed

Detection modes exposed: **MOG2** and **Frame Difference** (diff).

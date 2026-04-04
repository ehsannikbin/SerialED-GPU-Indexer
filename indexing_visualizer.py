import sys
import os
import re
import h5py
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QCheckBox, QGraphicsView, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsEllipseItem, 
                               QGraphicsTextItem, QGraphicsLineItem, QGroupBox)
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QImage, QPen, QColor, QFont, QBrush

# ==========================================
# Stream Parser
# ==========================================
class StreamParser:
    def __init__(self):
        # Default geometry values
        self.geometry = {'corner_x': 0.0, 'corner_y': 0.0, 'res': 1.0}

    def parse(self, filename):
        events = []
        current_event = {}
        in_chunk = False
        in_peaks = False
        in_reflections = False
        in_geom = False
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # --- 1. Parse Geometry Block (Only once) ---
                if line.startswith("----- Begin geometry file"):
                    in_geom = True
                    continue
                if line.startswith("----- End geometry file"):
                    in_geom = False
                    continue
                
                if in_geom:
                    if "p0/corner_x" in line: self.geometry['corner_x'] = float(line.split('=')[1].strip())
                    if "p0/corner_y" in line: self.geometry['corner_y'] = float(line.split('=')[1].strip())
                    if "res =" in line: self.geometry['res'] = float(line.split('=')[1].strip())
                    continue

                # --- 2. Parse Chunks ---
                if line.startswith("----- Begin chunk -----"):
                    in_chunk = True
                    current_event = {
                        'filename': None, 'event': None, 
                        'peaks': [], 'reflections': []
                    }
                
                elif line.startswith("----- End chunk -----"):
                    in_chunk = False
                    if current_event['filename']:
                        events.append(current_event)
                    current_event = {}

                if not in_chunk: continue
                
                # Metadata
                if line.startswith("Image filename:"): 
                    current_event['filename'] = line.split(": ")[1]
                elif line.startswith("Event:"): 
                    current_event['event'] = line.split(": ")[1]
                
                # Note: We ignore "header/float...shift" lines here because we will 
                # read the TRUE shifts from HDF5 later.

                # Peaks Section
                if line.startswith("Peaks from peak search"): in_peaks = True; continue
                elif line.startswith("End of peak list"): in_peaks = False; continue
                
                if in_peaks:
                    parts = line.split()
                    # Expecting: fs ss 1/d I panel
                    if len(parts) >= 2 and parts[0] != "fs/px":
                        try: 
                            current_event['peaks'].append((float(parts[0]), float(parts[1])))
                        except: pass

                # Reflections Section
                if line.startswith("Reflections measured"): in_reflections = True; continue
                elif line.startswith("End of reflections"): in_reflections = False; continue
                
                if in_reflections:
                    parts = line.split()
                    # Expecting: h k l I sigI peak bg fs ss panel
                    if len(parts) >= 9 and parts[0] != "h":
                        try: 
                            # Save (fs, ss, h, k, l)
                            current_event['reflections'].append((
                                float(parts[7]), float(parts[8]), 
                                int(parts[0]), int(parts[1]), int(parts[2])
                            ))
                        except: pass
        
        return events, self.geometry

# ==========================================
# Main Window
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PinkIndexer Results Viewer")
        self.resize(1200, 900)
        
        self.events = []
        self.geometry = {}
        self.current_idx = 0
        self.base_dir = "." 
        
        self.setup_ui()
        self.peak_items = []
        self.refl_items = []
        self.label_items = []
        self.center_items = []

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # --- Controls ---
        controls = QVBoxLayout()
        controls.setAlignment(Qt.AlignTop)
        
        self.btn_load = QPushButton("Load Stream File")
        self.btn_load.clicked.connect(self.load_stream)
        controls.addWidget(self.btn_load)
        
        self.grp_info = QGroupBox("Event Info")
        info_layout = QVBoxLayout()
        self.lbl_filename = QLabel("File: N/A")
        self.lbl_event = QLabel("Event: N/A")
        self.lbl_stats = QLabel("Peaks: 0 | Indexed: 0")
        self.lbl_center = QLabel("Center: N/A") # Debug info
        info_layout.addWidget(self.lbl_filename)
        info_layout.addWidget(self.lbl_event)
        info_layout.addWidget(self.lbl_stats)
        info_layout.addWidget(self.lbl_center)
        self.grp_info.setLayout(info_layout)
        controls.addWidget(self.grp_info)
        
        self.grp_view = QGroupBox("Visualization")
        view_layout = QVBoxLayout()
        self.chk_peaks = QCheckBox("Found Peaks (Red)")
        self.chk_peaks.setChecked(True)
        self.chk_peaks.toggled.connect(self.toggle_layers)
        
        self.chk_refl = QCheckBox("Indexed Spots (Green)")
        self.chk_refl.setChecked(True)
        self.chk_refl.toggled.connect(self.toggle_layers)
        
        self.chk_labels = QCheckBox("HKL Labels")
        self.chk_labels.setChecked(False)
        self.chk_labels.toggled.connect(self.toggle_layers)
        
        self.chk_center = QCheckBox("Beam Center (Yellow)")
        self.chk_center.setChecked(True)
        self.chk_center.toggled.connect(self.toggle_layers)

        view_layout.addWidget(self.chk_peaks)
        view_layout.addWidget(self.chk_refl)
        view_layout.addWidget(self.chk_labels)
        view_layout.addWidget(self.chk_center)
        self.grp_view.setLayout(view_layout)
        controls.addWidget(self.grp_view)
        
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("<< Prev")
        self.btn_prev.clicked.connect(self.prev_event)
        self.btn_next = QPushButton("Next >>")
        self.btn_next.clicked.connect(self.next_event)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        controls.addLayout(nav_layout)
        
        self.lbl_counter = QLabel("0 / 0")
        self.lbl_counter.setAlignment(Qt.AlignCenter)
        controls.addWidget(self.lbl_counter)
        layout.addLayout(controls, 1) 
        
        # --- Viewer ---
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        layout.addWidget(self.view, 4) 

    def load_stream(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Stream", ".", "Stream Files (*.stream)")
        if not fname: return
        
        self.base_dir = os.path.dirname(fname)
        parser = StreamParser()
        self.events, self.geometry = parser.parse(fname)
        
        if not self.events:
            self.lbl_filename.setText("No events found!")
            return
            
        self.current_idx = 0
        self.update_display()

    def prev_event(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()

    def next_event(self):
        if self.current_idx < len(self.events) - 1:
            self.current_idx += 1
            self.update_display()

    def get_hdf5_data(self, filename, event_str):
        """
        Reads Image Data AND Experimental Shifts from HDF5.
        Returns: (image_data, shift_x_mm, shift_y_mm)
        """
        full_path = filename
        if not os.path.isabs(full_path):
            full_path = os.path.join(self.base_dir, filename)
            
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            return None, 0, 0

        event_idx = 0
        if '//' in event_str:
            try: event_idx = int(event_str.replace('//', ''))
            except: pass
            
        img_data = None
        sx_mm, sy_mm = 0.0, 0.0
        
        try:
            with h5py.File(full_path, 'r') as f:
                # 1. Load Image
                if '/data' in f:
                    d = f['/data']
                    img_data = d[event_idx] if d.ndim == 3 else d[()]
                elif '/entry/data/data' in f:
                    d = f['/entry/data/data']
                    img_data = d[event_idx] if d.ndim == 3 else d[()]
                
                # 2. Load Experimental Shifts
                # Try common paths
                path_x = '/center/shift_x_mm'
                path_y = '/center/shift_y_mm'
                
                if path_x not in f: 
                    path_x = '/entry/data/shift_x_mm'
                    path_y = '/entry/data/shift_y_mm'
                
                if path_x in f:
                    dx = f[path_x]
                    dy = f[path_y]
                    sx_mm = float(dx[event_idx] if dx.ndim > 0 else dx[()])
                    sy_mm = float(dy[event_idx] if dy.ndim > 0 else dy[()])
                    
        except Exception as e:
            print(f"HDF5 Error: {e}")
            
        return img_data, sx_mm, sy_mm

    def update_display(self):
        if not self.events: return
        evt = self.events[self.current_idx]
        
        # 1. Load Data from HDF5
        img_data, shift_x_mm, shift_y_mm = self.get_hdf5_data(evt['filename'], evt['event'])
        
        # Update Labels
        self.lbl_filename.setText(f"File: {os.path.basename(evt['filename'])}")
        self.lbl_event.setText(f"Event: {evt['event']}")
        self.lbl_stats.setText(f"Peaks: {len(evt['peaks'])} | Indexed: {len(evt['reflections'])}")
        
        # Clear Scene
        self.scene.clear()
        self.peak_items, self.refl_items, self.label_items, self.center_items = [], [], [], []
        
        # 2. Draw Image
        if img_data is not None:
            # Log contrast
            img_log = np.log1p(np.maximum(img_data, 0))
            img_norm = (img_log / np.max(img_log) * 255).astype(np.uint8)
            h, w = img_norm.shape
            qimg = QImage(img_norm.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            self.scene.addItem(QGraphicsPixmapItem(pixmap))
            self.scene.setSceneRect(0, 0, w, h)
        else:
            self.scene.addText("Image Load Failed", QFont("Arial", 20, QFont.Bold)).setDefaultTextColor(Qt.red)
            return

        # 3. Draw Peaks (Red)
        pen_peak = QPen(QColor(255, 50, 50), 1)
        for fs, ss in evt['peaks']:
            item = QGraphicsEllipseItem(fs-4, ss-4, 8, 8)
            item.setPen(pen_peak)
            self.scene.addItem(item)
            self.peak_items.append(item)

        # 4. Draw Indexed Reflections (Green)
        pen_refl = QPen(QColor(50, 255, 50), 1)
        font = QFont("Arial", 8); font.setBold(True)
        
        for fs, ss, h, k, l in evt['reflections']:
            item = QGraphicsEllipseItem(fs-3, ss-3, 6, 6)
            item.setPen(pen_refl)
            self.scene.addItem(item)
            self.refl_items.append(item)
            
            txt = QGraphicsTextItem(f"{h},{k},{l}")
            txt.setPos(fs+5, ss+5)
            txt.setFont(font)
            txt.setDefaultTextColor(QColor(100, 255, 100))
            self.scene.addItem(txt)
            self.label_items.append(txt)

        # 5. Draw Beam Center (Yellow) - USING HDF5 SHIFTS
        # Constants from Geometry (Stream Header)
        cx_geom = -self.geometry.get('corner_x', 0)
        cy_geom = -self.geometry.get('corner_y', 0)
        res = self.geometry.get('res', 1.0)
        
        # Formula: Beam = -Corner - 0.5 - (Shift_mm * 1e-3 * Res)
        beam_x = cx_geom - 0.5 - (shift_x_mm * 1e-3 * res)
        beam_y = cy_geom - 0.5 - (shift_y_mm * 1e-3 * res)
        
        self.lbl_center.setText(f"Beam: ({beam_x:.1f}, {beam_y:.1f})")
        
        pen_cen = QPen(QColor(255, 255, 0), 2)
        l1 = QGraphicsLineItem(beam_x-20, beam_y, beam_x+20, beam_y)
        l2 = QGraphicsLineItem(beam_x, beam_y-20, beam_x, beam_y+20)
        l1.setPen(pen_cen)
        l2.setPen(pen_cen)
        self.scene.addItem(l1)
        self.scene.addItem(l2)
        self.center_items.extend([l1, l2])

        self.toggle_layers()

    def toggle_layers(self):
        for item in self.peak_items: item.setVisible(self.chk_peaks.isChecked())
        for item in self.refl_items: item.setVisible(self.chk_refl.isChecked())
        for item in self.label_items: item.setVisible(self.chk_labels.isChecked())
        for item in self.center_items: item.setVisible(self.chk_center.isChecked())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
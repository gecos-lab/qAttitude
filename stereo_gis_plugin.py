# -*- coding: utf-8 -*-
# qAttitude @ Andrea Bistacchi 2024-06-26

import os
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon

from .stereo_gis_dialog import StereoGisDialog


class StereoGisPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dlg = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        self.action = QAction(QIcon(icon_path), "qAttitude", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("qAttitude", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        if self.action:
            self.iface.removePluginMenu("qAttitude", self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None

    def run(self):
        if self.dlg is None:
            self.dlg = StereoGisDialog(self.iface)
        self.dlg.show()
        self.dlg.raise_()
        self.dlg.activateWindow()

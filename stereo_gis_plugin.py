# -*- coding: utf-8 -*-
# qAttitude @ Andrea Bistacchi 2024-06-26

from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon

from .stereo_gis_dialog import StereoGisDialog


class StereoGisPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dlg = None

    def initGui(self):
        self.action = QAction(QIcon(), "qAttitude", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("qAttitude", self.action)

    def unload(self):
        if self.action:
            self.iface.removePluginMenu("qAttitude", self.action)
            self.action = None

    def run(self):
        if self.dlg is None:
            self.dlg = StereoGisDialog(self.iface)
        self.dlg.show()
        self.dlg.raise_()
        self.dlg.activateWindow()

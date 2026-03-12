# -*- coding: utf-8 -*-
# qAttitude @ Andrea Bistacchi 2024-06-26

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QDockWidget
from qgis.core import Qgis
from qgis.PyQt.QtCore import Qt
import os.path
import pandas as pd
from .qAttitude_main import qAttitudeDialog


class qAttitudePlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = "&qAttitude"
        self.toolbar = self.iface.pluginToolBar()
        self.dock_widget = None

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None,
    ):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, "icon.svg")
        self.add_action(
            icon_path,
            text="qAttitude",
            callback=self.run,
            parent=self.iface.mainWindow(),
        )

    def onClosePlugin(self):
        self.unload()

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu("&qAttitude", action)
            self.iface.removeToolBarIcon(action)
        if self.dock_widget:
            self.iface.removeDockWidget(self.dock_widget)

    def run(self):
        if not self.dock_widget:
            self.dock_widget = QDockWidget("qAttitude", self.iface.mainWindow())
            self.dock_widget.setAllowedAreas(
                Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
            )
            self.dlg = qAttitudeDialog(self.iface, self)
            self.dock_widget.setWidget(self.dlg)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
        else:
            if self.dock_widget.isVisible():
                self.dock_widget.hide()
            else:
                self.dock_widget.show()

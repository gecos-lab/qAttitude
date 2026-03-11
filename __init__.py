# -*- coding: utf-8 -*-

def classFactory(iface):
    from .qAttitude_plugin import qAttitudePlugin
    return qAttitudePlugin(iface)

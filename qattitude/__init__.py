# -*- coding: utf-8 -*-
def classFactory(iface):
    from .qAttitude_core import qAttitudePlugin
    return qAttitudePlugin(iface)

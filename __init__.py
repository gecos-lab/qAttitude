# -*- coding: utf-8 -*-
def classFactory(iface):
    from .qAttitude import qAttitudePlugin
    return qAttitudePlugin(iface)

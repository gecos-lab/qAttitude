# -*- coding: utf-8 -*-
# qAttitude @ Andrea Bistacchi 2024-06-26

def classFactory(iface):
    from .stereo_gis_plugin import StereoGisPlugin

    return StereoGisPlugin(iface)

# qAttitude

qAttitude © 2026 by Andrea Bistacchi, released under GNU AGPLv3 license.

QGis plugin for on-the-fly orientation analysis of geological data, with stereoplots and orientation statistics. Runs under QGis 3 and 4 (tested with 3.44 and 4.0).

To install, first install the [PackageInstallerQgis](https://plugins.qgis.org/plugins/PackageInstallerQgis/) plugin by [BRGM](https://www.brgm.fr), available in QGis under Plugins > Manage and Install Plugins. This is used to manage required libraries for this and other plugins. Then clone the [qAttitude folder](https://github.com/gecos-lab/qAttitude) in your QGis plugins folder, which under Windows is C:\Users\<your user name>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins or C:\Users\<your user name>\AppData\Roaming\QGIS\QGIS4\profiles\default\python\plugins. Then you will find the *qAttitude* plugin under Plugins > Manage and Install Plugins and you will be able to activate it.

To run, just drag and drop a layer with orientation data, then if necessary adjust some options regarding planes vs. lines, dip/direction or plunge/trend fields, and axial/bidirectional vs. polar/unidirectional data (e.g. fold axes or foliatins are axial/bidirectionla while bedding with younging direction or slip directions are polar/unidirectional).

Different standard mean orientation statistics (Von Mises-Fisher, Kent, Bingham) are calculated and shown in the log window in the lower part of the plugin panel, and can be turned on or off in the plot with checkboxes.

K-means clustring can be performed changing the number of clusters (selecting just 1 cluster means "no clustering"), optionally picking with mouse clicks the cluster seed points. For many statistical and numerical reasons, clustering can fail for specific numbers of clusters, and in this case a warning is shown in the log and no cluster data is plotted. Cluster labels can be transferred back to the data layer to freeze the analysis results and define different structural domains.

Test data and a simple QGis project are provided in the test_data folder.


# qAttitude

qAttitude © 2025 by Andrea Bistacchi, released under GNU AGPLv3 license.

QGis plugin for on-the-fly orientation analysis of geological data, with stereoplots and orientation statistics. Runs under QGis 3 and 4 (tested with 3.44 and 4.0).

To install, just download (qattitude.zip)[https://github.com/gecos-lab/qAttitude/blob/main/qattitude.zip], then in QGis > Plugins > Manage and Install Plugins > Install from ZIP > select qattitude.zip > Install Plugin > Yes > Close, and you will find qAttitude in the Plugins menu and toolbar.

To run, just drag and drop a layer with orientation data, then if necessary adjust some options regarding planes vs. lines, dip/direction or plunge/trend fields, and axial/bidirectional vs. polar/unidirectional data (e.g. fold axes or foliatins are axial/bidirectionla while bedding with younging direction or slip directions are polar/unidirectional).

Different standard mean orientation statistics (Von Mises-Fisher, Kent, Bingham) are calculated and shown in the log window in the lower part of the plugin panel, and can be turned on or off in the plot with checkboxes.

K-means clustring can be performed changing the number of clusters (selecting just 1 cluster means "no clustering"), optionally picking with mouse clicks the cluster seed points. For many statistical and numerical reasons, clustering can fail for specific numbers of clusters, and in this case a warning is shown in the log and no cluster data is plotted. Cluster labels can be transferred back to the data layer to freeze the analysis results and define different structural domains.

Test data and a simple QGis project are provided in the test_data folder.

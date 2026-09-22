import scyjava as sj

# Import some key java classes
# A little bit about scyjava: we can import classes from Java and use them in python just as if they were in Java. Here are some classes that for TrackMate.
ThresholdToSelection = sj.jimport('ij.plugin.filter.ThresholdToSelection')
ImageProcessor = sj.jimport('ij.process.ImageProcessor')
RoiEnlarger = sj.jimport('ij.plugin.RoiEnlarger')
TmXmlWriter = sj.jimport('fiji.plugin.trackmate.io.TmXmlWriter')
Model = sj.jimport('fiji.plugin.trackmate.Model')
File = sj.jimport('java.io.File')
Settings = sj.jimport('fiji.plugin.trackmate.Settings')
TrackMate = sj.jimport('fiji.plugin.trackmate.TrackMate')
Logger = sj.jimport('fiji.plugin.trackmate.Logger')
LogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.LogDetectorFactory')
SparseLAPTrackerFactory = sj.jimport('fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory')
FeatureFilter = sj.jimport('fiji.plugin.trackmate.features.FeatureFilter')

# And a couple data types
JDouble = sj.jimport('java.lang.Double')
JInteger = sj.jimport('java.lang.Integer')
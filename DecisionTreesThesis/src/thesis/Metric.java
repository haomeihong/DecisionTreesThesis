package thesis;

import weka.core.Attribute;
import weka.core.Instances;

public abstract class Metric {
	public abstract boolean isMaximizingMetric();
	public abstract double computeMetric(Instances data, Attribute att) throws Exception;
}

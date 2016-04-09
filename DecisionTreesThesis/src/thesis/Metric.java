package thesis;

import java.io.Serializable;

import weka.core.Attribute;
import weka.core.Instances;

public abstract class Metric implements Serializable {
	public abstract boolean isMaximizingMetric();
	public abstract double computeMetric(Instances data, Attribute att) throws Exception;
	public abstract String getStr();
}

package thesis.metrics;

import thesis.CustomId3;
import thesis.Metric;
import weka.core.Attribute;
import weka.core.Instances;

public abstract class ImpurityMetric extends Metric {

	public abstract double computeImpurity(Instances data) throws Exception;
	
	@Override
	public double computeMetric(Instances data, Attribute att) throws Exception {
		double gain = computeImpurity(data);
	    Instances[] splitData = CustomId3.splitData(data, att);
	    for (int j = 0; j < att.numValues(); j++) {
	      if (splitData[j].numInstances() > 0) {
	        gain -= ((double) splitData[j].numInstances() /
	                     (double) data.numInstances()) *
	        		computeImpurity(splitData[j]);
	      }
	    }
	    return gain;
	}

	@Override
	public boolean isMaximizingMetric() {
		return true;
	}
	
	@Override
	public double valueMakeLeaf() {
		return 0.0;
	}
}
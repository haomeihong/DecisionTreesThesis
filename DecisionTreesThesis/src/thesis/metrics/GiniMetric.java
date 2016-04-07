package thesis.metrics;

import java.util.Enumeration;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class GiniMetric extends ImpurityMetric {

	private double computeGiniImpurity(Instances data) throws Exception {

		double[] classCounts = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}
		double gini = 0;
		for (int j = 0; j < data.numClasses(); j++) {
			if (classCounts[j] > 0) {
				double x = (classCounts[j] / (double) data.numInstances());
				gini += x*x;
			}
		}
		return 1-gini;
	}
	
	@Override
	public double computeImpurity(Instances data) throws Exception {
		return computeGiniImpurity(data);
	}

}

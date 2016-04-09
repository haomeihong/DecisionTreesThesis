package thesis.metrics;

import java.util.Enumeration;

import weka.core.Instance;
import weka.core.Instances;

public class AccuracyMetric extends ImpurityMetric {

	private double computeAccImpurity(Instances data) throws Exception {

		double[] classCounts = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}
		double minacc = 1.0;
		for (int j = 0; j < data.numClasses(); j++) {
			if (classCounts[j] > 0) {
				double acc = (classCounts[j] / (double) data.numInstances());
				if (acc < minacc)
					minacc = acc;
			}
		}
		return minacc;
	}
	
	@Override
	public double computeImpurity(Instances data) throws Exception {
		return computeAccImpurity(data);
	}

	@Override
	public String getStr() {
		return "Accuracy (purity)";
	}

}


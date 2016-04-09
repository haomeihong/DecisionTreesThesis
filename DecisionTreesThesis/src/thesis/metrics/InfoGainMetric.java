package thesis.metrics;

import java.util.Enumeration;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class InfoGainMetric extends ImpurityMetric {

	private double computeEntropy(Instances data) throws Exception {

		double[] classCounts = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}
		double entropy = 0;
		for (int j = 0; j < data.numClasses(); j++) {
			if (classCounts[j] > 0) {
				entropy -= classCounts[j] * Utils.log2(classCounts[j]);
			}
		}
		entropy /= (double) data.numInstances();
		return entropy + Utils.log2(data.numInstances());
	}

	@Override
	public double computeImpurity(Instances data) throws Exception {
		return computeEntropy(data);
	}

	@Override
	public String getStr() {
		return "Info Gain";
	}
}

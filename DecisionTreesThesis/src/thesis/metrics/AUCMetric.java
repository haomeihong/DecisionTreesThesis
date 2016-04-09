package thesis.metrics;

import java.util.Enumeration;

import thesis.CustomId3;
import thesis.Metric;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class AUCMetric extends Metric {
	@Override
	public double computeMetric(Instances data, Attribute att) throws Exception {
		
		if (data.numClasses() != 2)
			throw new Exception("AUCMetric only works with two classes!");
		
		double[] parentClassCounts = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			parentClassCounts[(int) inst.classValue()]++;
		}
		
		double weightedAUC = 0;
		
		Instances[] splitData = CustomId3.splitData(data, att);
		for (int j = 0; j < att.numValues(); j++) {
			if (splitData[j].numInstances() > 0) {
				
				double[] classCounts = new double[splitData[j].numClasses()];
				Enumeration instEnum2 = splitData[j].enumerateInstances();
				while (instEnum2.hasMoreElements()) {
					Instance inst = (Instance) instEnum2.nextElement();
					classCounts[(int) inst.classValue()]++;
				}
				
				int posIndex = Utils.maxIndex(classCounts);
				int negIndex = Utils.minIndex(classCounts);
				
				double TP = classCounts[posIndex];
				double FP = classCounts[negIndex];
				
				double P = parentClassCounts[posIndex];
				double N = parentClassCounts[negIndex];
				
				if (Utils.eq(N,0.0))
					return 1.0;
				
				double TPR = TP / P;
				double FPR = FP / N;
				
				double auc = .5 * (TPR - FPR + 1);
				
				weightedAUC += ((double) splitData[j].numInstances() / (double) data.numInstances()) * auc;
			}
		}	
		
		return weightedAUC;
	}

	@Override
	public boolean isMaximizingMetric() {
		return true;
	}

	@Override
	public String getStr() {
		return "Area Under the Curve";
	}

	@Override
	public double valueMakeLeaf() {
		return 1.0;
	}

}

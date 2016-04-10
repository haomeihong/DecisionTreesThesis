package thesis.experiments;

import java.util.Random;

import thesis.CustomId3;
import thesis.DataLoader;
import thesis.metrics.AUCMetric;
import thesis.metrics.AccuracyMetric;
import thesis.metrics.FMeasureMetric;
import thesis.metrics.GiniMetric;
import thesis.metrics.InfoGainMetric;
import weka.classifiers.Evaluation;
import weka.core.Instances;
 
public class Experiment1 {
 
	public static void main(String[] args) throws Exception {
		Instances data = DataLoader.loadData("car");
		
		System.out.println(data.toSummaryString());
		System.out.println(data.classAttribute().toString());
		System.out.println(data.attributeStats(data.numAttributes() - 1));

		CustomId3[] models = { 
				new CustomId3(new AccuracyMetric()),
				new CustomId3(new FMeasureMetric()),
				new CustomId3(new AUCMetric()),
				new CustomId3(new InfoGainMetric()),
				new CustomId3(new GiniMetric())
		};
		
		for (int j = 0; j < models.length; j++) {
			Evaluation evaluation = new Evaluation(data);
			evaluation.crossValidateModel(models[j], data, 10, new Random(1));
			System.out.println(models[j].getMetric().getStr());
			System.out.println(" | ACCURACY    = " + evaluation.pctCorrect());
			System.out.println(" | F-MEASURE   = " + evaluation.weightedFMeasure()*100);
			System.out.println(" | AUC         = " + evaluation.weightedAreaUnderROC()*100);
		}
	}
}

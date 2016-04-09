package test;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;

import thesis.CustomId3;
import thesis.metrics.AccuracyMetric;
import thesis.metrics.GiniMetric;
import thesis.metrics.InfoGainMetric;
import weka.classifiers.Evaluation;
import weka.core.Instances;
 
public class WekaTest {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile("datasets/car.arff");
 
		Instances data = new Instances(datafile);
		for (int i = 0; i < data.numAttributes(); i++)
			data.deleteWithMissing(i);
		int classIndex = data.numAttributes() - 1; 
		data.setClassIndex(classIndex);
	
		System.out.println(data.toSummaryString());
		System.out.println(data.classAttribute().toString());
		System.out.println(data.attributeStats(classIndex));

		CustomId3[] models = { 
				new CustomId3(new InfoGainMetric()),
				new CustomId3(new GiniMetric()),
				new CustomId3(new AccuracyMetric())
		};
		
		for (int j = 0; j < models.length; j++) {
			Evaluation evaluation = new Evaluation(data);
			evaluation.crossValidateModel(models[j], data, 10, new Random(1));
			System.out.println(models[j].getMetric().getStr());
			System.out.println(" | ACCURACY   = " + evaluation.pctCorrect());
			System.out.println(" | AUC        = " + evaluation.weightedAreaUnderROC());
			System.out.println(" | F-MEASURE  = " + evaluation.weightedFMeasure());
			System.out.println(" | PRECISION  = " + evaluation.weightedPrecision());
			System.out.println(" | RECALL     = " + evaluation.weightedRecall());
		}
	}
}

package test;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;

import thesis.CustomId3;
import thesis.metrics.GiniMetric;
import thesis.metrics.InfoGainMetric;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
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
 
	public static Evaluation classify(Classifier model,
			Instances trainingSet, Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);
 
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
 
		return evaluation;
	}
 
	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
 
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
 
		
		Random rand = new Random(System.currentTimeMillis()); 
		Instances randData = new Instances(data);  
		randData.randomize(rand);    
	
		//randData.stratify(numberOfFolds);
		 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = randData.trainCV(numberOfFolds, i);
			split[1][i] = randData.testCV(numberOfFolds, i);
		}
 
		return split;
	}
 
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile("datasets/car.arff");
 
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		
		System.out.println(data.toSummaryString());
		
		//data.setClassIndex(0);
 
		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data, 10);
 
		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
 
		// Use a set of classifiers
		CustomId3[] models = { 
				new CustomId3(new InfoGainMetric()),
				new CustomId3(new GiniMetric())
		};
		
		for (int j = 0; j < models.length; j++) {
 
			Evaluation evaluation = new Evaluation(data);
			evaluation.crossValidateModel(models[j], data, 10, new Random(1));
			System.out.println(models[j].getMetric().getStr());
			System.out.println(" | ACC = " + evaluation.pctCorrect());
			System.out.println(" | ROC = " + evaluation.areaUnderROC(0));
			System.out.println(" | F-M = " + evaluation.fMeasure(0));

		}
 
	}
}

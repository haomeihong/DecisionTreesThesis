package test;

import java.io.File;
import java.io.IOException;
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
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class test {

	public static Instances readFile(String fileName) {
		CSVLoader loader = new CSVLoader();
		try {
			loader.setFile(new File(fileName));
			Instances data = loader.getDataSet();
			for (int i = 0; i < data.numAttributes(); i++)
				data.deleteWithMissing(i);
			data.setClassIndex(0); // first attribute is class
			
			/*ArffSaver s = new ArffSaver();
			s.setInstances(data);
			s.setFile(new File("datasets/mushroom.arff"));
			s.writeBatch();*/
			
			return data;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
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
		 //randData.randomize(rand);    
	
		 //randData.stratify(numberOfFolds);
		
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = randData.trainCV(numberOfFolds, i);
			split[1][i] = randData.testCV(numberOfFolds, i);
		}
 
		return split;
	}
	
	public static void main(String[] args) throws Exception {
		Instances data = readFile("datasets/mushroom.data");
		

		
		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data, 10);

		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];

		// Use a set of classifiers
		Classifier[] models = { 
			new Id3(),
			new CustomId3(new InfoGainMetric()),
			new CustomId3(new GiniMetric())
		};
	
		/*Evaluation evaluation = new Evaluation(data);
		evaluation.crossValidateModel(models[0], data, 10, new Random(100));
		System.out.println(evaluation.toSummaryString());*/

		// Run for each model
		for (int j = 0; j < models.length; j++) {

			// Collect every group of predictions for current model in a
			// FastVector
			FastVector predictions = new FastVector();

			// For each training-testing split pair, train and test the
			// classifier
			for (int i = 0; i < trainingSplits.length; i++) {
				Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);

				predictions.appendElements(validation.predictions());

				// Uncomment to see the summary for each training-testing pair.
				// System.out.println(models[j].toString());
			}

			// Calculate overall accuracy of current classifier on all splits
			double accuracy = calculateAccuracy(predictions);

			// Print current classifier's name and accuracy in a complicated,
			// but nice-looking way.
			System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": "
					+ String.format("%.2f%%", accuracy) + "\n---------------------------------");
		}
	}

}

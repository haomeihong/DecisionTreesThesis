package thesis;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class DataLoader {
	private static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		return inputReader;
	}
	
	public static Instances loadData(String datasetName) throws IOException {
		BufferedReader datafile = readDataFile("datasets/"+datasetName+".arff");
		Instances data = new Instances(datafile);

		// Last attribute is the class
		data.setClassIndex(data.numAttributes() - 1);
		
		// Delete all instances with missing values
		for (int i = 0; i < data.numAttributes(); i++)
			data.deleteWithMissing(i);
		
		makeBinary(data);
		
		return data;
	}
	
	private static void makeBinary(Instances data) {
		Attribute classAttribute = data.classAttribute();
		String positiveClass = classAttribute.value(0);
		String negativeClass = "NOT_" + positiveClass;

		FastVector classValues = new FastVector();
		classValues.addElement(positiveClass);
		classValues.addElement(negativeClass);
		Attribute newClassAttribute = new Attribute("THE_CLASS",classValues);
		data.insertAttributeAt(newClassAttribute, data.numAttributes());
		
		for (int i = 0; i < data.numInstances(); i++) {
			Instance instance = data.instance(i);
			String classValue = instance.stringValue(classAttribute);
			if (classValue.equals(positiveClass)) {
				instance.setValue(data.numAttributes() - 1, positiveClass);
			} else {
				instance.setValue(data.numAttributes() - 1, negativeClass);
			}
		}
		data.setClassIndex(data.numAttributes() - 1); // make new class attribute
		data.deleteAttributeAt(data.numAttributes() - 2); // delete old class attribute
	}
}

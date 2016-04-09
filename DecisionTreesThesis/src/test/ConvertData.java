package test;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class ConvertData {
	public static void main(String[] args) throws Exception {
		CSVLoader loader = new CSVLoader();
		try {
			loader.setFile(new File("datasets/monks-1.data"));
			loader.setNominalAttributes("1-7");
			Instances data = loader.getDataSet();
			for (int i = 0; i < data.numAttributes(); i++)
				data.deleteWithMissing(i);
			data.setClassIndex(0); // first attribute is class
			
			
			data.deleteAttributeAt(data.numAttributes() - 1);
			
			
			ArffSaver s = new ArffSaver();
			s.setInstances(data);
			s.setFile(new File("datasets/monks-1.arff"));
			s.writeBatch();
			
			System.out.println(data.toSummaryString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}

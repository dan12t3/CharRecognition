import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.List;
import java.util.Random;


/**
 * Daniyal Usman
 * 5152590
 * Iyuehan Yang
 * 5300231
 * COSC 4P80 - Convolutional neural network to classify handwritten recognition (Letters A, B, C)
 */



public class Evaluate {
    /**
     * height is the height we will normalise the image to
     */
    public static int height = 100;
    /**
     * width is the width we will normalise the image to
     */
    public static int width = 100;
    /**
     * channels equals the depth of the input layer. In this we have 3 due to the RGB values
     */
    public static int channels = 3;

    /**
     * batchSize tells us how many samples will be trained on before updating the weights.
     */
    public static int batchSize = 10;
    /**
     * seed is a number used for the random generator to keep everything consistent when testing other parameters
     */
    public static long seed = 3000;
    /**
     * rand is the random generator
     */
    public static Random rand = new Random(seed);
    /**
     * dataSize is the total number of samples we have in our system which is used for training and testing
     */
    public static int dataSize = 55*3;
    /**
     * classificationSize is the total number of classifications the final output will be categorized
     */
    public static int classificationSize = 3;
    /**
     * trainingDataSplit is the percentage of data that will be split for training and testing the system
     */

    /***
     * Main method which will open and load a saved convolutional neural network model and classify all the testing data.
     * @param args0
     * @throws Exception
     */
    public static void main(String[] args0) throws Exception{




        ParentPathLabelGenerator labels = new ParentPathLabelGenerator();
        File dataFolder = new File("src/main/resources/data/");
        FileSplit categorizedData = new FileSplit(dataFolder, NativeImageLoader.ALLOWED_FORMATS, rand);



        //change later on
        BalancedPathFilter filter = new BalancedPathFilter(rand, labels, dataSize, classificationSize, batchSize);


        //
        InputSplit[] dataSplit = categorizedData.sample(filter, 1,0);



        File savedNetwork = new File("src/main/resources/model.bin");
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(savedNetwork);

        //restored.params();

        //System.out.println(restored.params());
        //System.out.println(restored.get);


        ImageRecordReader imageReader = new ImageRecordReader(height, width, channels, labels);
        DataSetIterator iterator = new RecordReaderDataSetIterator(imageReader, batchSize, 1, classificationSize);

        DataNormalization normalize = new ImagePreProcessingScaler(0,1);

        Evaluate eval = new Evaluate(imageReader, dataSplit[0], iterator, classificationSize, normalize, restored);


    }

    /***
     * Constructor which will evaluate the testing data onto the trained convolutional neural network.
     * @param imageReader
     * @param testData
     * @param iterator
     * @param numLabels
     * @param normalise
     * @param network
     * @throws Exception
     */

    public Evaluate(ImageRecordReader imageReader, InputSplit testData, DataSetIterator iterator, int numLabels, DataNormalization normalise, MultiLayerNetwork network) throws Exception{

        imageReader.initialize(testData, null);

        normalise.fit(iterator);
        iterator.setPreProcessor(normalise);
        Evaluation eval = network.evaluate(iterator);

        System.out.println(eval.stats(true));

        // Example on how to get predict results with trained model
        iterator.reset();
        DataSet testDataSet = iterator.next();
        String expectedResult = testDataSet.getLabelName(0);
        List<String> predict = network.predict(testDataSet);
        String modelResult = predict.get(0);
        System.out.println("--------");

        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n");


    }
}

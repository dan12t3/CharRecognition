import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Daniyal Usman
 * 5152590
 * Iyuehan Yang
 * 5300231
 * COSC 4P80 - Convolutional neural network to classify handwritten recognition (Letters A, B, C)
 */
public class Train {

    /**
     * networkModel is the variable determining which convolutional model we will use to test the system
     */
    public static String networkModel="AlexNet";
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
     * iterations is a count for the number of iterations used for training
     */
    public static int iterations = 1;
    /**
     * batchSize tells us how many samples will be trained on before updating the weights.
     */
    public static int batchSize = 20;
    /**
     * epochs is the total number of epochs the system will train for.
     */
    public static int epochs = 60;
    /**
     * cores is the total number of GPU cores the system will use to train
     */
    public static int cores = 4;
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
    public static double trainingDataSplit = 0.5;
    /**
     * save is a boolean that will save our model
     */
    public static boolean save = true;

    /***
     * Main method which will run the core operation of training the convolutional neural network
     * @param args0
     * @throws Exception
     */

    public static void main(String[] args0) throws Exception{

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File dataFolder = new File("src/main/resources/data/");

        FileSplit categorizedData = new FileSplit(dataFolder, NativeImageLoader.ALLOWED_FORMATS, rand);

        BalancedPathFilter filter = new BalancedPathFilter(rand, labelMaker, dataSize, classificationSize, batchSize);

        InputSplit[] dataSplit = categorizedData.sample(filter, trainingDataSplit, 1-trainingDataSplit);
        InputSplit trainingData = dataSplit[0]; //Data used for training
        InputSplit testingData = dataSplit[1]; //Data used for testing

        //adding different transformations to our data set to create noise
        ImageTransform flipImage = new FlipImageTransform(rand); //adds a flipping transformation
        ImageTransform flipImage2 = new FlipImageTransform(new Random(seed+100)); //adds another flipping transformation
        ImageTransform warpImage = new WarpImageTransform(rand,42);//Adds a warping transformation


        List<ImageTransform> extraData = Arrays.asList(new ImageTransform[]{flipImage,flipImage2,warpImage});

        //normalizing data

        DataNormalization normalize = new ImagePreProcessingScaler(0,1);


        //Determining which network model the system will use based on the users input.
        MultiLayerNetwork network;
        switch (networkModel) {
            case "LeNet":
                network = NetworkModels.lenetModel(seed, iterations, channels, classificationSize, height, width);
                break;
            case "AlexNet":
                network = NetworkModels.alexnetModel(seed, iterations, channels, classificationSize, height, width);
                break;
            case "custom":
                network = NetworkModels.customModel(seed, iterations, channels, classificationSize, height, width);
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }

        network.init();
        network.setListeners(new ScoreIterationListener(1));

        ImageRecordReader imageReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator iterator;
        MultipleEpochsIterator trainingIterator;


        System.out.println("Train model....");
        // Train without transformations
        imageReader.initialize(trainingData, null);
        iterator = new RecordReaderDataSetIterator(imageReader, batchSize, 1, classificationSize);
        normalize.fit(iterator);
        iterator.setPreProcessor(normalize);
        trainingIterator = new MultipleEpochsIterator(epochs, iterator, cores);
        network.fit(trainingIterator);

        // Train with transformations
        for (ImageTransform transform : extraData) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            imageReader.initialize(trainingData, transform);
            iterator = new RecordReaderDataSetIterator(imageReader, batchSize, 1, classificationSize);
            normalize.fit(iterator);
            iterator.setPreProcessor(normalize);
            trainingIterator = new MultipleEpochsIterator(epochs, iterator, cores);
            network.fit(trainingIterator);
        }

        new Evaluate(imageReader, testingData, iterator, classificationSize, normalize,network);

        if (save) {
            System.out.println("Save model....");
            String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
            ModelSerializer.writeModel(network, basePath + "model.bin", true);
        }

        //Evaluate network







    }
}

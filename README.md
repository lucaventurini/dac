DAC
=======
DAC is a Distributed Associative Classifier built on Apache Spark.

The algorithm builds multiple associative classifiers in parallel, finally combining them in a single model by exploiting [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating).

There are three main parameters to set:
* the number of models to train in parallel, `numModels`
* the fraction of dataset to sample for each model `sampleSize` and 
* the minimum support threshold `minSupport`.


## Usage in a Spark Application

This section assumes you are already confident with writing a Spark application.
If not, go to Standalone usage.

The example below shows how to load a comma-separated file and train a DAC model with the given parameters.
The test error is calculated to measure the algorithm accuracy.
Note that each record is an array of `Long`, and the least integer in the row is used as class label, while all the other items are used as features.

```scala
import it.polito.dbdmg.ml.DACEnsemble

// Load and parse the data file.
val inf = sc.textFile(inputFile)
val data = inf.map(_.split(",").map(_.toLong))
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a DAC model.
val numModels = 10
val numClasses = 3
val minSupport = 0.1
val sampleSize = 0.1

val model = new DACEnsemble(numModels = numModels, numClasses = numClasses, minSupport = minSupport, sampleSize = sampleSize)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  label = point.min //predict uses the least integer in the row as class label,
  val prediction = model.predict(point) //so be sure your record is structured accordingly
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println("Test Error = " + testErr)
println("Learned DAC model:\n" + model.toDebugString)
```

## Standalone usage
### Compile

The project uses sbt to compile & build the DAC tool. The following command downloads the dependencies, compiles the code, and creates the required jar file in the `target` directory.

	sbt package

The generated jar file, containing DAC, is in `target/scala-2.10/dac_2.10-0.2.jar`

The jar file dac_2.10-0.2.jar includes also all the needed dependencies. Hence, no libraries need to be uploaded on the Hadoop cluster in order to run DAC.

Please note that the implementation is developed and tested on 2.5.0-cdh5.3.1 Cloudera configuration.

### Input dataset format

The input dataset for the class DacUciCv is a list of records with n+1 columns separated by comma, where n is the number of features and the last column is the class label of the record.
The first line can be optionally an header, marked by an initial `|`.
The following is an example extracted from the IRIS dataset converted to this format and discretized, which is available online at https://www.sgi.com/tech/mlc/db/ together with other UCI datasets:

    |Discretization into bins of size 0
    5\.85-Inf, 3\.05-Inf, 2\.6-4\.85, 0\.75-1\.65, Iris-versicolor.
    5\.45-5\.85, 3\.05-Inf, -Inf-2\.6, -Inf-0\.75, Iris-setosa.
    5\.85-Inf, 3\.05-Inf, 4\.85-Inf, 1\.65-Inf, Iris-virginica.
    5\.85-Inf, -Inf-3\.05, 2\.6-4\.85, 0\.75-1\.65, Iris-versicolor.


### Run

The class DacUciCv executes a 10-fold cross-validation of the DAC algorithm on a given input dataset, and outputs the results of the cross-validation on stdout.

To run it, be sure of having a discretized dataset in the above-mentioned format, let's say `iris.data`, and then run:

    spark-submit --master local[*] --class DacUciCv dac_2.10-0.2.jar irisd.data 10 0.1 0.1

where the last three parameters are, in order, the number of models to generate, the fraction of dataset to sample for each model and the minimum support threshold.

Of course the parameters of `spark-submit` change depending on your cluster configuration. Here we have set a local run with all the cores available (`--master local[*]`).

	
## References

The associative classifier where DAC builds upon is inspired to L3, presented in:
> Elena Baralis, Silvia Chiusano, Paolo Garza: A Lazy Approach to Associative Classification. IEEE Trans. Knowl. Data Eng. 20(2): 156-171 (2008). Available at: http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=4358963
	
## Credits

The code uses part of [MLlib](http://spark.apache.org/mllib/) code, provided under Apache License, Version 2.0.
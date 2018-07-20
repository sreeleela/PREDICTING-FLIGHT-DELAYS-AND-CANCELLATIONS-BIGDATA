import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.ml.feature.{RFormula, StringIndexer}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.sql.functions._
import org.apache.log4j.LogManager
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}

object Flights {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Application")

    val spark = SparkSession.builder().config(conf).getOrCreate()


    val manualSchema = new StructType(Array(
      new StructField("YEAR",IntegerType,true),
      new StructField("MONTH",IntegerType,true),
      new StructField("DAY",IntegerType,true),
      new StructField("DAY_OF_WEEK",IntegerType,true),
      new StructField("AIRLINE",StringType,true),
      new StructField("FLIGHT_NUMBER",IntegerType,true),
      new StructField("TAIL_NUMBER",StringType,true),
      new StructField("ORIGIN_AIRPORT",StringType,true),
      new StructField("DESTINATION_AIRPORT",StringType,true),
      new StructField("SCHEDULED_DEPARTURE",IntegerType,true),
      new StructField("DEPARTURE_TIME",IntegerType,true),
      new StructField("DEPARTURE_DELAY",IntegerType,true),
      new StructField("DEPARTURE_DELAY_INDEX",IntegerType,true),
      new StructField("TAXI_OUT",IntegerType,true),
      new StructField("WHEELS_OFF",IntegerType,true),
      new StructField("SCHEDULED_TIME",IntegerType,true),
      new StructField("ELAPSED_TIME",IntegerType,true),
      new StructField("AIR_TIME",IntegerType,true),
      new StructField("DISTANCE",IntegerType,true),
      new StructField("WHEELS_ON",IntegerType,true),
      new StructField("TAXI_IN",IntegerType,true),
      new StructField("SCHEDULED_ARRIVAL",IntegerType,true),
      new StructField("ARRIVAL_TIME",IntegerType,true),
      new StructField("ARRIVAL_DELAY",IntegerType,true),
      new StructField("ARRIVAL_DELAY_INDEX",IntegerType,true),
      new StructField("DIVERTED",IntegerType,true),
      new StructField("CANCELLED",IntegerType,true),
      new StructField("CANCELLATION_REASON",IntegerType,true),
      new StructField("AIR_SYSTEM_DELAY",IntegerType,true),
      new StructField("SECURITY_DELAY",IntegerType,true),
      new StructField("AIRLINE_DELAY",IntegerType,true),
      new StructField("LATE_AIRCRAFT_DELAY",IntegerType,true),
      new StructField("WEATHER_DELAY",IntegerType,true)
    ))
    val path = "flights.csv"
    val data = spark.read.format("csv").schema(manualSchema).option("header","true").option("inferSchema","true").load(path)
    data.show(50)
    var fp, tp,tn,fn = 0.0

    print("Random Forest")
    //RF formulae
    val cancellationFormula = new RFormula().setFormula("CANCELLED ~ AIRLINE+FLIGHT_NUMBER+DESTINATION_AIRPORT+SCHEDULED_DEPARTURE+DISTANCE")
    val fittedRF = cancellationFormula.fit(data)
    val preparedDF = fittedRF.transform(data)
    //Spliting data
    val splits = preparedDF.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    //Random Forest
    val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setFeatureSubsetStrategy("auto")//.setMaxBins(30).setMaxDepth(10).setNumTrees(10).setSeed(4305).setImpurity("gini").setFeatureSubsetStrategy("auto")
    val model = rf.fit(trainingData)
    print(model.toDebugString)
    model.write.overwrite().save("RandomForestModel/Cancellations/model")
    var predictions = model.transform(testData)
    //predictions.toDF().show(900)

    val binarizer: Binarizer = new Binarizer().
      setInputCol("prediction").
      setOutputCol("binarized_prediction").
      setThreshold(0.5)
    val predictionBinary = binarizer.transform(predictions)
    //predictionBinary.select("label","features","prediction","binarized_prediction").show
    val wrongPredictions = predictionBinary.where(expr("label != binarized_prediction"))
    //wrongPredictions.select("label","features","prediction","binarized_prediction").show

    val countErrors = wrongPredictions.groupBy("label").agg(count("prediction").alias("Errors"))
    countErrors.show()
    fp = 0.0
    fn = 0.0
    var checkfp = countErrors.select("Errors").where("label = 0.0")
    if(!(checkfp.head(1).isEmpty)){fp = countErrors.select("Errors").where("label = 0.0").head().getLong(0)}
    if((checkfp.head(1).isEmpty)){fp = 0}
    var checkfn = countErrors.select("Errors").where("label = 1.0")
    if(!(checkfn.head(1).isEmpty)){fn = countErrors.select("Errors").where("label = 1.0").head().getLong(0)}
    if((checkfn.head(1).isEmpty)){fn = 0}
    println("False Positive(fp): "+fp)
    println("False Negative(fn): "+fn)

    val correctPredictions = predictionBinary.where(expr("label == binarized_prediction"))
    val countCorrectPredictions = correctPredictions.groupBy("label").agg(count("prediction").alias("Correct"))
    countCorrectPredictions.show()
    tp = 0.0
    tn = 0.0
    var checktp = countCorrectPredictions.select("Correct").where("label = 1.0")
    if(!(checktp.head(1).isEmpty)){tp = countCorrectPredictions.select("Correct").where("label = 1.0").head().getLong(0)}
    if((checktp.head(1).isEmpty)){tp = 0}
    var checktn = countCorrectPredictions.select("Correct").where("label = 0.0")
    if(!(checktn.head(1).isEmpty)){tn = countCorrectPredictions.select("Correct").where("label = 0.0").head().getLong(0)}
    if((checktn.head(1).isEmpty)){tn = 0}
    println("True Positive(tp): "+tp)
    println("True Negative(tn): "+tn)
    //Confusion Matrix
    val confusionMatrix = spark.sparkContext.parallelize(Seq((tp,fp),(fn,tn)))
    val cm = spark.createDataFrame(confusionMatrix).toDF("y_is_0","y_is_1")
    cm.write.mode("overwrite").save("RandomForestModel/Cancellations/confusionMatrix")
    cm.show()
    println("Random Forest Cancellation Accuracy: "+((tp+tn)/(tp+tn+fp+fn))*100)
    println("-----------------------------------------------------------------------------------------")
    //RF formulae
    val delayFormula = new RFormula().setFormula("DEPARTURE_DELAY_INDEX ~ AIRLINE+FLIGHT_NUMBER+DESTINATION_AIRPORT+SCHEDULED_DEPARTURE")
    val dFittedRF = delayFormula.fit(data)
    val dPreparedDF = dFittedRF.transform(data)
    //Spliting data
    val dSplits = dPreparedDF.randomSplit(Array(0.7, 0.3))
    val (dTrainingData, dTestData) = (dSplits(0), dSplits(1))
    //Random Forest
    val dRF = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setFeatureSubsetStrategy("auto")//.setMaxBins(30).setMaxDepth(10).setNumTrees(10).setSeed(4305).setImpurity("gini").setFeatureSubsetStrategy("auto")
    val dModel = dRF.fit(dTrainingData)
    print(dModel.toDebugString)
    dModel.write.overwrite().save("RandomForestModel/DepartureDelays/model")
    var dPredictions = dModel.transform(dTestData)
    //dPredictions.toDF().show(900)

    val dBinarizer: Binarizer = new Binarizer().
      setInputCol("prediction").
      setOutputCol("binarized_prediction").
      setThreshold(0.5)
    val dPredictionBinary = dBinarizer.transform(dPredictions)
    //dPredictionBinary.select("label","features","prediction","binarized_prediction").show
    val dWrongPredictions = dPredictionBinary.where(expr("label != binarized_prediction"))
    //dWrongPredictions.select("label","features","prediction","binarized_prediction").show

    val dCountErrors = dWrongPredictions.groupBy("label").agg(count("prediction").alias("Errors"))
    dCountErrors.show()
    fp = 0.0
    fn = 0.0
    checkfp = dCountErrors.select("Errors").where("label = 1.0")
    if(!(checkfp.head(1).isEmpty)){fp = dCountErrors.select("Errors").where("label = 1.0").head().getLong(0)}
    if((checkfp.head(1).isEmpty)){fp = 0}
    checkfn = dCountErrors.select("Errors").where("label = 0.0")
    if(!(checkfn.head(1).isEmpty)){fn = dCountErrors.select("Errors").where("label = 0.0").head().getLong(0)}
    if((checkfn.head(1).isEmpty)){fn = 0}
    println("False Positive(fp): "+fp)
    println("False Negative(fn): "+fn)

    val dCorrectPredictions = dPredictionBinary.where(expr("label == binarized_prediction"))
    val dCountCorrectPredictions = dCorrectPredictions.groupBy("label").agg(count("prediction").alias("Correct"))
    dCountCorrectPredictions.show()
    tp = 0.0
    tn = 0.0
    checktp = dCountCorrectPredictions.select("Correct").where("label = 0.0")
    if(!(checktp.head(1).isEmpty)){tp = dCountCorrectPredictions.select("Correct").where("label = 0.0").head().getLong(0)}
    if((checktp.head(1).isEmpty)){tp = 0}
    checktn = dCountCorrectPredictions.select("Correct").where("label = 1.0")
    if(!(checktn.head(1).isEmpty)){tn = dCountCorrectPredictions.select("Correct").where("label = 1.0").head().getLong(0)}
    if((checktn.head(1).isEmpty)){tn = 0}
    println("True Positive(tp): "+tp)
    println("True Negative(tn): "+tn)
    //Confusion Matrix
    val dConfusionMatrix = spark.sparkContext.parallelize(Seq((tp,fp),(fn,tn)))
    val dcm = spark.createDataFrame(dConfusionMatrix).toDF("y_is_0","y_is_1")
    dcm.write.mode("overwrite").save("RandomForestModel/DepartureDelays/confusionMatrix")
    dcm.show()
    println("Random Forest Departure delay accuracy: "+((tp+tn)/(tp+tn+fp+fn))*100)
    println("-----------------------------------------------------------------------------------------")
    //RF formulae
    val arrivalFormula = new RFormula().setFormula("ARRIVAL_DELAY_INDEX~AIRLINE+FLIGHT_NUMBER+ORIGIN_AIRPORT+SCHEDULED_ARRIVAL")
    val aFittedRF = arrivalFormula.fit(data)
    val aPreparedDF = aFittedRF.transform(data)
    //Spliting data
    val aSplits = aPreparedDF.randomSplit(Array(0.7, 0.3))
    val (aTrainingData, aTestData) = (aSplits(0), aSplits(1))
    //Random Forest
    val aRF = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setFeatureSubsetStrategy("auto")//.setMaxBins(30).setMaxDepth(10).setNumTrees(10).setSeed(4305).setImpurity("gini").setFeatureSubsetStrategy("auto")
    val aModel = aRF.fit(aTrainingData)
    print(aModel.toDebugString)
    aModel.write.overwrite().save("RandomForestModel/ArrivalDelays/model")
    var aPredictions = aModel.transform(aTestData)
    //dPredictions.toDF().show(900)

    val aBinarizer: Binarizer = new Binarizer().
      setInputCol("prediction").
      setOutputCol("binarized_prediction").
      setThreshold(0.5)
    val aPredictionBinary = aBinarizer.transform(aPredictions)
    //dPredictionBinary.select("label","features","prediction","binarized_prediction").show
    val aWrongPredictions = aPredictionBinary.where(expr("label != binarized_prediction"))
    //dWrongPredictions.select("label","features","prediction","binarized_prediction").show

    val aCountErrors = aWrongPredictions.groupBy("label").agg(count("prediction").alias("Errors"))
    aCountErrors.show()
    fp = 0.0
    fn = 0.0
    checkfp = aCountErrors.select("Errors").where("label = 1.0")
    if(!(checkfp.head(1).isEmpty)){fp = aCountErrors.select("Errors").where("label = 1.0").head().getLong(0)}
    if((checkfp.head(1).isEmpty)){fp = 0}
    checkfn = aCountErrors.select("Errors").where("label = 0.0")
    if(!(checkfn.head(1).isEmpty)){fn = aCountErrors.select("Errors").where("label = 0.0").head().getLong(0)}
    if((checkfn.head(1).isEmpty)){fn = 0}
    println("False Positive(fp): "+fp)
    println("False Negative(fn): "+fn)

    val aCorrectPredictions = aPredictionBinary.where(expr("label == binarized_prediction"))
    val aCountCorrectPredictions = aCorrectPredictions.groupBy("label").agg(count("prediction").alias("Correct"))
    aCountCorrectPredictions.show()
    tp = 0.0
    tn = 0.0
    checktp = aCountCorrectPredictions.select("Correct").where("label = 0.0")
    if(!(checktp.head(1).isEmpty)){tp = aCountCorrectPredictions.select("Correct").where("label = 0.0").head().getLong(0)}
    if((checktp.head(1).isEmpty)){tp = 0}
    checktn = aCountCorrectPredictions.select("Correct").where("label = 1.0")
    if(!(checktn.head(1).isEmpty)){tn = aCountCorrectPredictions.select("Correct").where("label = 1.0").head().getLong(0)}
    if((checktn.head(1).isEmpty)){tn = 0}
    println("True Positive(tp): "+tp)
    println("True Negative(tn): "+tn)
    //Confusion Matrix
    val aConfusionMatrix = spark.sparkContext.parallelize(Seq((tp,fp),(fn,tn)))
    val acm = spark.createDataFrame(aConfusionMatrix).toDF("y_is_0","y_is_1")
    acm.write.mode("overwrite").save("RandomForestModel/ArrivalDelays/confusionMatrix")
    acm.show()
    println("Random Forest Arrival delay accuracy: "+((tp+tn)/(tp+tn+fp+fn))*100)
   println("-----------------------------------------------------------------------------------------")
    println("Logistic Regression")
  //RF formulae
    val cLFormula = new RFormula().setFormula("CANCELLED ~ AIRLINE+FLIGHT_NUMBER+DESTINATION_AIRPORT+SCHEDULED_DEPARTURE+DISTANCE")
    val lFittedRF = cLFormula.fit(data)
    val lPreparedDF = lFittedRF.transform(data)
    //Spliting data
    val lSplits = lPreparedDF.randomSplit(Array(0.7, 0.3))
    val (lTrainingData, lTestData) = (lSplits(0), lSplits(1))
    //Logistic Regression
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val lModel = lr.fit(lTrainingData)
    lModel.write.overwrite().save("LogisticRegressionModel/Cancellations/model")
    var lPredictions = lModel.transform(lTestData)
    //lPredictions.toDF().show(900)

    val lbinarizer: Binarizer = new Binarizer().
      setInputCol("prediction").
      setOutputCol("binarized_prediction").
      setThreshold(0.5)
    val lPredictionBinary = lbinarizer.transform(lPredictions)
    //predictionBinary.select("label","features","prediction","binarized_prediction").show
    val lWrongPredictions = lPredictionBinary.where(expr("label != binarized_prediction"))
    //wrongPredictions.select("label","features","prediction","binarized_prediction").show

    val lCountErrors = lWrongPredictions.groupBy("label").agg(count("prediction").alias("Errors"))
    lCountErrors.show()
    fp = 0.0
    fn = 0.0
    var lcheckfp = lCountErrors.select("Errors").where("label = 0.0")
    if(!(lcheckfp.head(1).isEmpty)){fp = lCountErrors.select("Errors").where("label = 0.0").head().getLong(0)}
    if((lcheckfp.head(1).isEmpty)){fp = 0}
    var lcheckfn = lCountErrors.select("Errors").where("label = 1.0")
    if(!(lcheckfn.head(1).isEmpty)){fn = lCountErrors.select("Errors").where("label = 1.0").head().getLong(0)}
    if((lcheckfn.head(1).isEmpty)){fn = 0}
    println("False Positive(fp): "+fp)
    println("False Negative(fn): "+fn)

    val lCorrectPredictions = lPredictionBinary.where(expr("label == binarized_prediction"))
    val lCountCorrectPredictions = lCorrectPredictions.groupBy("label").agg(count("prediction").alias("Correct"))
    lCountCorrectPredictions.show()
    tp = 0.0
    tn = 0.0
    var lchecktp = lCountCorrectPredictions.select("Correct").where("label = 1.0")
    if(!(lchecktp.head(1).isEmpty)){tp = lCountCorrectPredictions.select("Correct").where("label = 1.0").head().getLong(0)}
    if((lchecktp.head(1).isEmpty)){tp = 0}
    var lchecktn = lCountCorrectPredictions.select("Correct").where("label = 0.0")
    if(!(lchecktn.head(1).isEmpty)){tn = lCountCorrectPredictions.select("Correct").where("label = 0.0").head().getLong(0)}
    if((lchecktn.head(1).isEmpty)){tn = 0}
    println("True Positive(tp): "+tp)
    println("True Negative(tn): "+tn)
    //Confusion Matrix
    val lConfusionMatrix = spark.sparkContext.parallelize(Seq((tp,fp),(fn,tn)))
    val lcm = spark.createDataFrame(lConfusionMatrix).toDF("y_is_0","y_is_1")
    lcm.write.mode("overwrite").save("LogisticRegressionModel/Cancellations/confusionMatrix")
    lcm.show()
    println("LR Cancellation Accuracy: "+((tp+tn)/(tp+tn+fp+fn))*100)
    println("-----------------------------------------------------------------------------------------")
    println("Logistic Regression")
    //RF formulae
    val dcLFormula = new RFormula().setFormula("DEPARTURE_DELAY_INDEX ~ AIRLINE+FLIGHT_NUMBER+DESTINATION_AIRPORT+SCHEDULED_DEPARTURE")
    val dlFittedRF = dcLFormula.fit(data)
    val dlPreparedDF = dlFittedRF.transform(data)
    //Spliting data
    val dlSplits = dlPreparedDF.randomSplit(Array(0.7, 0.3))
    val (dlTrainingData, dlTestData) = (dlSplits(0), dlSplits(1))
    //Logistic Regression
    val dlr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val dlModel = dlr.fit(dlTrainingData)
    dlModel.write.overwrite().save("LogisticRegressionModel/DepartureDelays/model")
    var dlPredictions = dlModel.transform(dlTestData)
    //lPredictions.toDF().show(900)

    val dlbinarizer: Binarizer = new Binarizer().
      setInputCol("prediction").
      setOutputCol("binarized_prediction").
      setThreshold(0.5)
    val dlPredictionBinary = dlbinarizer.transform(dlPredictions)
    //predictionBinary.select("label","features","prediction","binarized_prediction").show
    val dlWrongPredictions = dlPredictionBinary.where(expr("label != binarized_prediction"))
    //wrongPredictions.select("label","features","prediction","binarized_prediction").show

    val dlCountErrors = dlWrongPredictions.groupBy("label").agg(count("prediction").alias("Errors"))
    dlCountErrors.show()
    fp = 0.0
    fn = 0.0
    var dlcheckfp = dlCountErrors.select("Errors").where("label = 1.0")
    if(!(dlcheckfp.head(1).isEmpty)){fp = dlCountErrors.select("Errors").where("label = 1.0").head().getLong(0)}
    if((dlcheckfp.head(1).isEmpty)){fp = 0}
    var dlcheckfn = dlCountErrors.select("Errors").where("label = 0.0")
    if(!(dlcheckfn.head(1).isEmpty)){fn = dlCountErrors.select("Errors").where("label = 0.0").head().getLong(0)}
    if((dlcheckfn.head(1).isEmpty)){fn = 0}
    println("False Positive(fp): "+fp)
    println("False Negative(fn): "+fn)

    val dlCorrectPredictions = dlPredictionBinary.where(expr("label == binarized_prediction"))
    val dlCountCorrectPredictions = dlCorrectPredictions.groupBy("label").agg(count("prediction").alias("Correct"))
    dlCountCorrectPredictions.show()
    tp = 0.0
    tn = 0.0
    var dlchecktp = dlCountCorrectPredictions.select("Correct").where("label = 0.0")
    if(!(dlchecktp.head(1).isEmpty)){tp = dlCountCorrectPredictions.select("Correct").where("label = 0.0").head().getLong(0)}
    if((dlchecktp.head(1).isEmpty)){tp = 0}
    var dlchecktn = dlCountCorrectPredictions.select("Correct").where("label = 1.0")
    if(!(dlchecktn.head(1).isEmpty)){tn = dlCountCorrectPredictions.select("Correct").where("label = 1.0").head().getLong(0)}
    if((dlchecktn.head(1).isEmpty)){tn = 0}
    println("True Positive(tp): "+tp)
    println("True Negative(tn): "+tn)
    //Confusion Matrix
    val dlConfusionMatrix = spark.sparkContext.parallelize(Seq((tp,fp),(fn,tn)))
    val dlcm = spark.createDataFrame(dlConfusionMatrix).toDF("y_is_0","y_is_1")
    dlcm.write.mode("overwrite").save("LogisticRegressionModel/DepartureDelays/confusionMatrix")
    dlcm.show()
    println("LR Departure Delays Accuracy: "+((tp+tn)/(tp+tn+fp+fn))*100)
    println("-----------------------------------------------------------------------------------------")
    //RF formulae
    val acLFormula = new RFormula().setFormula("ARRIVAL_DELAY_INDEX~AIRLINE+FLIGHT_NUMBER+ORIGIN_AIRPORT+SCHEDULED_ARRIVAL")
    val alFittedRF = acLFormula.fit(data)
    val alPreparedDF = alFittedRF.transform(data)
    //Spliting data
    val alSplits = alPreparedDF.randomSplit(Array(0.7, 0.3))
    val (alTrainingData, alTestData) = (alSplits(0), alSplits(1))
    //Logistic Regression
    val alr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val alModel = alr.fit(alTrainingData)
    alModel.write.overwrite().save("LogisticRegressionModel/ArrivalDelays/model")
    var alPredictions = alModel.transform(alTestData)
    //lPredictions.toDF().show(900)

    val albinarizer: Binarizer = new Binarizer().
      setInputCol("prediction").
      setOutputCol("binarized_prediction").
      setThreshold(0.5)
    val alPredictionBinary = albinarizer.transform(alPredictions)
    //predictionBinary.select("label","features","prediction","binarized_prediction").show
    val alWrongPredictions = alPredictionBinary.where(expr("label != binarized_prediction"))
    //wrongPredictions.select("label","features","prediction","binarized_prediction").show

    val alCountErrors = alWrongPredictions.groupBy("label").agg(count("prediction").alias("Errors"))
    alCountErrors.show()
    fp = 0.0
    fn = 0.0
    var alcheckfp = alCountErrors.select("Errors").where("label = 1.0")
    if(!(alcheckfp.head(1).isEmpty)){fp = alCountErrors.select("Errors").where("label = 1.0").head().getLong(0)}
    if((alcheckfp.head(1).isEmpty)){fp = 0}
    var alcheckfn = alCountErrors.select("Errors").where("label = 0.0")
    if(!(alcheckfn.head(1).isEmpty)){fn = alCountErrors.select("Errors").where("label = 0.0").head().getLong(0)}
    if((alcheckfn.head(1).isEmpty)){fn = 0}
    println("False Positive(fp): "+fp)
    println("False Negative(fn): "+fn)

    val alCorrectPredictions = alPredictionBinary.where(expr("label == binarized_prediction"))
    val alCountCorrectPredictions = alCorrectPredictions.groupBy("label").agg(count("prediction").alias("Correct"))
    alCountCorrectPredictions.show()
    tp = 0.0
    tn = 0.0
    var alchecktp = alCountCorrectPredictions.select("Correct").where("label = 0.0")
    if(!(alchecktp.head(1).isEmpty)){tp = alCountCorrectPredictions.select("Correct").where("label = 0.0").head().getLong(0)}
    if((alchecktp.head(1).isEmpty)){tp = 0}
    var alchecktn = alCountCorrectPredictions.select("Correct").where("label = 1.0")
    if(!(alchecktn.head(1).isEmpty)){tn = alCountCorrectPredictions.select("Correct").where("label = 1.0").head().getLong(0)}
    if((alchecktn.head(1).isEmpty)){tn = 0}
    println("True Positive(tp): "+tp)
    println("True Negative(tn): "+tn)
    //Confusion Matrix
    val alConfusionMatrix = spark.sparkContext.parallelize(Seq((tp,fp),(fn,tn)))
    val alcm = spark.createDataFrame(alConfusionMatrix).toDF("y_is_0","y_is_1")
    alcm.write.mode("overwrite").save("LogisticRegressionModel/ArrivalDelays/confusionMatrix")
    alcm.show()
    println("LR Arrival Delays Accuracy: "+((tp+tn)/(tp+tn+fp+fn))*100)
  }
}
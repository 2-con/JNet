package tensor.tools;

public class Scalers {
  public static double[] standard(double[] array) {
    double mean = Statistics.mean(array);
    double stdDev = Statistics.stdev(array);

    return ArrayTools.foreach(new double[array.length], n -> (n - mean) / stdDev);
  }

  public static double[] minMax(double[] array) {
    double max = Statistics.max(array);
    double min = Statistics.min(array);

    return ArrayTools.foreach(new double[array.length], n -> (n - min) / (max - min));
  }

  public static double[] robust(double[] array) {
    double median = Statistics.median(array);
    double iqr = Statistics.iqr(array);

    return ArrayTools.foreach(new double[array.length], n -> (n - median) / iqr);
  }

  public static double[] maxAbs(double[] array) {
    double maxAbs = Statistics.max(ArrayTools.foreach(array, n -> Math.abs(n)));

    return ArrayTools.foreach(new double[array.length], n -> n / maxAbs);
  }
}

package tensor.tools;

import java.util.Arrays;
import java.util.HashMap;

public class Statistics {
  public static double sum(double[] array) {
    double ans = 0;
    for (double n : array) ans += n;
    return ans;
  }

  public static double min(double[] array) {
    double ans = Double.MAX_VALUE;
    for (double n : array) ans = (n < ans) ? n : ans;
    return ans;
  }

  public static double max(double[] array) {
    double ans = Double.MIN_VALUE;
    for (double n : array) ans = (n > ans) ? n : ans;
    return ans;
  }

  public static double mean(double[] array) {
    return ArrayTools.sum(array) / array.length;
  }
  
  public static double median(double[] array) {
    double[] copy = array.clone();
    Arrays.sort(copy);
    return copy[array.length/2];
  }

  public static double mode(double[] array) {
    HashMap<Double, Integer> dict = new HashMap<>();
    
    // count
    for (double n : array) {
      if (dict.containsKey(n)) {
        dict.put(n, dict.get(n) + 1);
      } else {
        dict.put(n, 1);
      }
    }

    // find max
    int max = 0;
    double ans = 0;
    for (HashMap.Entry<Double, Integer> entry : dict.entrySet()) {
      if (entry.getValue() > max) {
        max = entry.getValue();
        ans = entry.getKey();
      }
    }

    return ans;
  }
  
  public static double range(double[] array) {
    return max(array) - min(array);
  }

  public static double variance(double[] array) {
    double mu = mean(array);
    double[] shifted = ArrayTools.foreach(new double[array.length], n -> Math.pow(n - mu, 2));

    return mean(shifted);
  }

  public static double stdev(double[] array) {
    return Math.sqrt(variance(array));
  }

  public static double skew(double[] array) {
    double mu = mean(array);
    double stdeviation = stdev(array);

    double[] shifted = ArrayTools.foreach(new double[array.length], n -> Math.pow(n - mu, 3));

    return mean(shifted) / Math.pow(stdeviation, 3);
  }
  
  public static double kurtosis(double[] array) {
    double mu = mean(array);
    double stdeviation = stdev(array);

    double[] shifted = ArrayTools.foreach(new double[array.length], n -> Math.pow(n - mu, 4));

    return mean(shifted) / Math.pow(stdeviation, 4);
  }
  
  public static double quartile1(double[] array) {
    double[] copy = array.clone();
    Arrays.sort(copy);
    return copy[array.length/4];
  }
  
  public static double quartile3(double[] array) {
    double[] copy = array.clone();
    Arrays.sort(copy);
    return copy[(array.length/4) * 3];
  }

  public static double iqr(double[] array) {
    return quartile3(array) - quartile1(array);
  }

  // multi-input

  public static double percentile(double[] array, double percentile) {
    double[] sortedData = array.clone();
    Arrays.sort(sortedData);

    int index = (int) Math.ceil(percentile / 100.0 * sortedData.length) - 1;
    return sortedData[Math.max(0, index)];
  }
}

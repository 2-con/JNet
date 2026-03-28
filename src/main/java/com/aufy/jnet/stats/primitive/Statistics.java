package com.aufy.jnet.stats.primitive;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.aufy.jnet.tensor.core.backend.util.ArrayTools;

public class Statistics {
  public static double prod(double[] array) {
    double ans = 1;
    for (double n : array) ans *= n;
    return ans;
  }

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
    return sum(array) / array.length;
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

  public static double l1Norm(double[] array) {
    double ans = 0;
    for (double n : array) ans += Math.abs(n);
    return ans;
  }

  public static double frobeniusNorm(double[] array) {
    double ans = 0;
    for (double n : array) ans += n * n;
    return Math.sqrt(ans);
  }

  public static double percentile(double[] array, double percentile) {
    double[] sortedData = array.clone();
    Arrays.sort(sortedData);

    int index = (int) Math.ceil(percentile / 100.0 * sortedData.length) - 1;
    return sortedData[Math.max(0, index)];
  }

  // int arrays

  public static int prod(int[] array) {
    int ans = 1;
    for (double n : array) ans *= n;
    return ans;
  }

  public static int sum(int[] array) {
    int ans = 0;
    for (int n : array) ans += n;
    return ans;
  }

  public static int min(int[] array) {
    int ans = Integer.MAX_VALUE;
    for (int n : array) ans = Math.min(n, ans);
    return ans;
  }

  public static int max(int[] array) {
    int ans = Integer.MIN_VALUE;
    for (int n : array) ans = Math.max(n, ans);
    return ans;
  }

  public static double mean(int[] array) {
    return (double) sum(array) / array.length;
  }

  public static int median(int[] array) {
    int[] copy = array.clone();
    Arrays.sort(copy);
    return copy[array.length / 2];
  }

  public static int mode(int[] array) {
    HashMap<Integer, Integer> dict = new HashMap<>();
    for (int n : array) {
      dict.put(n, dict.getOrDefault(n, 0) + 1);
    }
    int maxCount = 0;
    int ans = array[0];
    for (Map.Entry<Integer, Integer> entry : dict.entrySet()) {
      if (entry.getValue() > maxCount) {
        maxCount = entry.getValue();
        ans = entry.getKey();
      }
    }
    return ans;
  }

  public static int range(int[] array) {
    return max(array) - min(array);
  }

  public static double variance(int[] array) {
    double mu = mean(array);
    double sumSqDiff = 0;
    for (int n : array) sumSqDiff += Math.pow(n - mu, 2);
    return sumSqDiff / array.length;
  }

  public static double stdev(int[] array) {
    return Math.sqrt(variance(array));
  }

  public static double skew(int[] array) {
    double mu = mean(array);
    double sigma = stdev(array);
    double sumCubedDiff = 0;
    for (int n : array) sumCubedDiff += Math.pow(n - mu, 3);
    return (sumCubedDiff / array.length) / Math.pow(sigma, 3);
  }

  public static double kurtosis(int[] array) {
    double mu = mean(array);
    double sigma = stdev(array);
    double sumFourthDiff = 0;
    for (int n : array) sumFourthDiff += Math.pow(n - mu, 4);
    return (sumFourthDiff / array.length) / Math.pow(sigma, 4);
  }

  public static int quartile1(int[] array) {
    int[] copy = array.clone();
    Arrays.sort(copy);
    return copy[array.length / 4];
  }

  public static int quartile3(int[] array) {
    int[] copy = array.clone();
    Arrays.sort(copy);
    return copy[(array.length / 4) * 3];
  }

  public static int iqr(int[] array) {
    return quartile3(array) - quartile1(array);
  }

  public static int l1Norm(int[] array) {
    int ans = 0;
    for (int n : array) ans += Math.abs(n);
    return ans;
  }

  public static double frobeniusNorm(int[] array) {
    double ans = 0;
    for (int n : array) ans += (double) n * n;
    return Math.sqrt(ans);
  }

  public static int percentile(int[] array, double percentile) {
    int[] sortedData = array.clone();
    Arrays.sort(sortedData);
    int index = (int) Math.ceil(percentile / 100.0 * sortedData.length) - 1;
    return sortedData[Math.max(0, index)];
  }

}

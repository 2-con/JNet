package com.aufy.jnet.tensor.core.backend.compute;

import java.util.Arrays;
import java.util.stream.IntStream;

import com.aufy.jnet.stats.primitive.Statistics;

public class Generator {
  /*
  complex intializers go in math or nn
  */
  
  public static double[] fill(double value, int... shape) {
    int size = Statistics.prod(shape);
    double[] data = new double[size];
    Arrays.fill(data, value);

    return data;
  }

  public static double[] zeros(int... shape) {
    int size = Statistics.prod(shape);
    
    return new double[size];
  }

  public static double[] ones(int... shape) {
    int size = Statistics.prod(shape);
    double[] data = new double[size];
    Arrays.fill(data, 1.0);

    return data;
  }

  public static int[] arrange(int size) {
    return IntStream.range(0, size + 1).toArray();
  }
}

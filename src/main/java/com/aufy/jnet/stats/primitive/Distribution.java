package com.aufy.jnet.stats.primitive;

import java.util.Random;

public interface Distribution {
  private static final Random RANDOM = new Random();

  public static double[] generateGaussian(int... shape) {
    int size = Statistics.prod(shape);

    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextGaussian();
    }
    return data;
  }

  public static double[] generateExponential(int... shape) {
    int size = Statistics.prod(shape);

    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextExponential();
    }
    return data;
  }

  public static double[] generateUniform(int... shape) {
    int size = Statistics.prod(shape);

    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextDouble();
    }
    return data;
  }
}

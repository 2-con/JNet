package com.aufy.jnet.stats.primitive;

import java.util.Random;

public interface Distribution {
  public static Random generator = new Random();

  public static double[] generateGaussian(int... shape) {
    int size = Statistics.prod(shape);

    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = generator.nextGaussian();
    }
    return data;
  }

  public static double[] generateExponential(int... shape) {
    int size = Statistics.prod(shape);

    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = generator.nextExponential();
    }
    return data;
  }

  public static double[] generateUniform(int... shape) {
    int size = Statistics.prod(shape);

    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = generator.nextDouble();
    }
    return data;
  }
}

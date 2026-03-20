package tensor.core;

import java.util.Random;

public class Generator {
  private static final Random RANDOM = new Random();

  public static double[] generateGaussian(int size) {
    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextGaussian();
    }
    return data;
  }

  public static double[] generateExponential(int size) {
    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextExponential();
    }
    return data;
  }

  public static double[] generateUniform(int size) {
    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextDouble();
    }
    return data;
  }
}

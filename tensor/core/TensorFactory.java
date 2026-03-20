package tensor.core;

import java.util.Random;

public class TensorFactory {
  private static final Random RANDOM = new Random();

  public static double[] generateRandom(int size) {
    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextGaussian();
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

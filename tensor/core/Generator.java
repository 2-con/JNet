package tensor.core;

import java.util.Arrays;
import java.util.Random;

public class Generator {
  private static final Random RANDOM = new Random();

  public static double[] generateGaussian(int... shape) {
    int size = Engine.sizeOf(shape);

    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextGaussian();
    }
    return data;
  }

  public static double[] generateExponential(int... shape) {
    int size = Engine.sizeOf(shape);

    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextExponential();
    }
    return data;
  }

  public static double[] generateUniform(int... shape) {
    int size = Engine.sizeOf(shape);

    double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = RANDOM.nextDouble();
    }
    return data;
  }

  public static double[] zeros(int... shape) {
    int size = Engine.sizeOf(shape);
    
    return new double[size];
  }

  public static double[] ones(int... shape) {
    int size = Engine.sizeOf(shape);
    double[] data = new double[size];
    Arrays.fill(data, 1.0);

    return data;
  }


}

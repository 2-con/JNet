package tensor.tools;

import java.util.Arrays;
import tensor.core.impl.DataContainer;
import tensor.core.impl.TensorCore;

public class Debug {
  public static void parse(DataContainer tensor) {
    double[] data = tensor.dump();
    for (double d : data) {
      if (Double.isNaN(d)) throw new RuntimeException("Detected NaN in DataContaner");
      if (Double.isInfinite(d)) System.err.println("Warning: Infinity detected");
    }
  }

  public static void parse(TensorCore tensor) {
    double[] data = tensor.dump();
    for (double d : data) {
      if (Double.isNaN(d)) throw new RuntimeException("Detected NaN in DataContaner");
      if (Double.isInfinite(d)) System.err.println("Warning: Infinity detected");
    }
  }

  public static double sparsity(DataContainer tensor) {
    double[] data = tensor.dump();
    return (double) Arrays.stream(data).filter(d -> d == 0).count() / data.length;
  }

  public static double sparsity(TensorCore tensor) {
    double[] data = tensor.dump();
    return (double) Arrays.stream(data).filter(d -> d == 0).count() / data.length;
  }
}

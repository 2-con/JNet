package tensor.tools;

import java.util.function.DoubleUnaryOperator;

public class ArrayTools {

  // utilities

  public static double[] foreach(double[] array, DoubleUnaryOperator operation) {
    double[] copy = array.clone();

    for (int i = 0; i < copy.length; i++) {
      copy[i] = operation.applyAsDouble(copy[i]);
    }

    return copy;
  }

  // reductions

  public static double sum(double[] array) {
    double ans = 0;
    for (double n : array) ans += n;
    return ans;
  }

  public static double prod(double[] array) {
    double ans = 1;
    for (double n : array) ans *= n;
    return ans;
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
}
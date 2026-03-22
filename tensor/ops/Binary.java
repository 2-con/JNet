package tensor.ops;

@FunctionalInterface
public interface Binary {
  double _apply(double a, double b);

  /**
   * Applies a binary operation to all elements of an array.
   * 
   * @param data an array of double values
   * @param operation a binary operation to apply to each element, can either be a lambda expression or a reference to a method
   * @return a new array containing the results of applying the operation to each element of the input array
   */
  static double[] apply(double[] a, double[] b, Binary operation) {
    if (a.length != b.length) {
      throw new IllegalArgumentException(
        "Tensor size mismatch: Cannot operate on arrays of size " + a.length + " and " + b.length
      );
    }

    double[] result = new double[a.length];
    for (int i = 0; i < a.length; i++) {
      result[i] = operation._apply(a[i], b[i]);
    }
    return result;
  }
}

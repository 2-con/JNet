package tensor;

import java.util.Arrays;
import tensor.operation.Binary;
import tensor.operation.Unary;

public class Tensor {
  private final double[] data;
  private final double[] grad;
  private final int[] dims;
  private final int[] strides;

  private Tensor(double[] data, int[] dims) {
    this.data = data;
    this.grad = new double[data.length];
    this.dims = dims;
    this.strides = calculateStrides(dims);
  }

  // TENSOR UNARY OPERATIONS

  /**
   * Applies a unary operation to all elements of the tensor.
   * 
   * @param operation a unary operation to apply to each element, can either be a lambda expression or a reference to a method
   * @return a new tensor containing the results of applying the operation to each element of the input tensor
   */
  public Tensor map(Unary operation) {
    double[] resultData = Unary.apply(this.data, operation);
    return new Tensor(resultData, this.dims); // Wrap the result in a new Tensor
  }

  // TENSOR BINARY OPERATIONS
  
  /**
   * Combines two tensors with a binary operation.
   * 
   * @param other the other tensor to combine with
   * @param operation the binary operation to apply to each pair of elements
   * @return a new tensor containing the results of applying the operation to each pair of elements
   * @throws RuntimeException if the shapes of the two tensors are incompatible for binary operation
   */
  private Tensor combine(Tensor other, Binary operation) {
    if (!Arrays.equals(this.dims, other.dims)) {
      throw new RuntimeException("Incompatible shapes for Binary operation");
    }

    double[] resultData = Binary.apply(this.data, other.data, operation);

    return new Tensor(resultData, this.dims);
  }

  public Tensor add(Tensor other) {
    return combine(other, (a, b) -> a + b);
  }

  // CREATE TENSORS
  
  public static Tensor zeros(int... dims) {
    int size = 1;
    for (int d : dims) size *= d;
    return new Tensor(new double[size], dims);
  }
  
  // GET AND SET
  
  public double get(int... indices) {
    return data[getIndex(indices)];
  }
  
  public void set(double value, int... indices) {
    data[getIndex(indices)] = value;
  }
  
  private int getIndex(int... indices) {
    int index = 0;
    for (int i = 0; i < indices.length; i++) {
      index += indices[i] * strides[i];
    }
    return index;
  }

  // UTILITY

  private static int[] calculateStrides(int[] dims) {
    int[] strides = new int[dims.length];
    int st = 1;
    for (int i = dims.length - 1; i >= 0; i--) {
        strides[i] = st;
        st *= dims[i];
    }
    return strides;
  }

  public int[] shape() {
    return dims.clone();
  }

  @Override
  public String toString() {
    return "Tensor" + Arrays.toString(dims) + " Data: " + Arrays.toString(data);
  }
}
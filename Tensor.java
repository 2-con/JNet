import java.util.Arrays;
import tensor.core.Engine;
import tensor.core.Generator;
import tensor.operation.Binary;
import tensor.operation.Unary;

public class Tensor {
  public final double[] flat;
  public final int[] dims;
  public final int dim;
  private final double[] grad;
  private final int[] strides;
  
  private Tensor(double[] data, int[] dims) {
    this.flat = data;
    this.grad = new double[data.length];
    this.dims = dims;
    this.strides = Engine.calculateStrides(dims);
    this.dim = dims.length;
  }

  // TENSOR CONSTRUCTORS

  /**
   * Reshapes a flat array of doubles/ints/long into a Tensor.
   * 
   * @param data the flat array of doubles to reshape
   * @param dims the dimensions of the tensor
   * @return a new Tensor with the given dimensions and data
   * @throws IllegalArgumentException if the size of the data array does not match the dimensions given
   */
  public static Tensor tensorize(double[] data, int... dims) {
    int expectedSize = 1;
    for (int d : dims) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to dimentions " + Arrays.toString(dims));
    }
    // We clone the data to ensure the Tensor is truly independent (Immutability)
    return new Tensor(data.clone(), dims);
  }

  /**
   * Reshapes a flat array of doubles/ints/long into a Tensor.
   * 
   * @param data the flat array of doubles to reshape
   * @param dims the dimensions of the tensor
   * @return a new Tensor with the given dimensions and data
   * @throws IllegalArgumentException if the size of the data array does not match the dimensions given
   */
  public static Tensor tensorize(int[] data, int... dims) {
    double[] doublesArray = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      doublesArray[i] = (double) data[i];
    }

    int expectedSize = 1;
    for (int d : dims) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to dimentions " + Arrays.toString(dims));
    }

    // We clone the data to ensure the Tensor is truly independent (Immutability)
    return new Tensor(doublesArray.clone(), dims);
  }

  /**
   * Reshapes a flat array of doubles/ints/long into a Tensor.
   * 
   * @param data the flat array of doubles to reshape
   * @param dims the dimensions of the tensor
   * @return a new Tensor with the given dimensions and data
   * @throws IllegalArgumentException if the size of the data array does not match the dimensions given
   */
  public static Tensor tensorize(long[] data, int... dims) {
    double[] doublesArray = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      doublesArray[i] = (double) data[i];
    }

    int expectedSize = 1;
    for (int d : dims) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to dimentions " + Arrays.toString(dims));
    }

    // We clone the data to ensure the Tensor is truly independent (Immutability)
    return new Tensor(doublesArray.clone(), dims);
  }

  /**
   * Returns a Tensor with random values uniformly distributed between 0 and 1.
   * 
   * @param dims the dimensions of the Tensor
   * @return a new Tensor with the given dimensions and uniformly distributed values
   */
  public static Tensor randomUniform(int... dims) {
    int size = 1;
    for (int d : dims) size *= d;
    double[] data = Generator.generateUniform(size);
    return new Tensor(data, dims);
  }

  /**
   * Returns a Tensor with random values drawn from a standard normal distribution.
   * 
   * @param dims the dimensions of the Tensor
   * @return a new Tensor with the given dimensions and standard normally distributed values
   */
  public static Tensor randomNormal(int... dims) {
    int size = 1;
    for (int d : dims) size *= d;
    double[] data = Generator.generateGaussian(size);
    return new Tensor(data, dims);
  }

  /**
   * Returns a Tensor with random values drawn from an exponential distribution.
   * 
   * @param dims the dimensions of the Tensor
   * @return a new Tensor with the given dimensions and exponentially distributed values
   */
  public static Tensor randomExponential(int... dims) {
    int size = 1;
    for (int d : dims) size *= d;
    double[] data = Generator.generateExponential(size);
    return new Tensor(data, dims);
  }

  /**
   * Returns a Tensor with all elements set to zero.
   * 
   * @param dims the dimensions of the Tensor
   * @return a new Tensor with the given dimensions and all elements set to zero
   */
  public static Tensor zeros(int... dims) {
    int size = 1;
    for (int d : dims) size *= d;
    return new Tensor(new double[size], dims);
  }

  /**
   * Returns a Tensor with all elements set to zero, with the same dimensions as the given tensor.
   * 
   * @param tensor the tensor to return a zeros Tensor for
   * @return a new Tensor with the same dimensions as the given tensor and all elements set to zero
   */
  public static Tensor zerosLike(Tensor tensor) {
    return zeros(tensor.dims);
  }

  /**
   * Returns a Tensor with all elements set to one.
   * 
   * @param dims the dimensions of the Tensor
   * @return a new Tensor with the given dimensions and all elements set to one
   */
  public static Tensor ones(int... dims) {
    int size = 1;
    for (int d : dims) size *= d;
    double[] data = new double[size];
    Arrays.fill(data, 1.0);
    return new Tensor(data, dims);
  }

  /**
   * Returns a Tensor with all elements set to one, with the same dimensions as the given tensor.
   * 
   * @param tensor the tensor to return a ones Tensor for
   * @return a new Tensor with the same dimensions as the given tensor and all elements set to one
   */
  public static Tensor onesLike(Tensor tensor) {
    return ones(tensor.dims);
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // STATIC TENSOR OPERATIONS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * Applies a unary operation to all elements of the given tensor.
   *
   * @param tensor the input tensor
   * @param operation the unary operation to apply
   * @return a new tensor containing the results
   */
  public static Tensor apply(Tensor tensor, Unary operation) {
    double[] resultData = Unary.apply(tensor.flat, operation);
    return new Tensor(resultData, tensor.dims);
  }

  /**
   * Adds a constant value to all elements of the given tensor.
   *
   * @param tensor the input tensor
   * @param value the constant value to add
   * @return a new tensor containing the results
   */
  public static Tensor add(Tensor tensor, double value) {
    return apply(tensor, x -> x + value);
  }

  /**
   * Multiplies all elements of the given tensor by a constant value.
   *
   * @param tensor the input tensor
   * @param value the constant value to multiply by
   * @return a new tensor containing the results
   */
  public static Tensor mul(Tensor tensor, double value) {
    return apply(tensor, x -> x * value);
  }

  // BINARY

  /**
   * Combines two tensors with a binary operation elementwise.
   *
   * @param a the first tensor
   * @param b the second tensor
   * @param operation the binary operation to apply
   * @return a new tensor containing the results
   * @throws RuntimeException if the shapes of the two tensors are mismatched
   */
  public static Tensor combine(Tensor a, Tensor b, Binary operation) {
    if (!Arrays.equals(a.dims, b.dims)) {
      throw new RuntimeException("Incompatible shapes for Binary operation; got " + Arrays.toString(a.dims) + " and " + Arrays.toString(b.dims));
    }
    double[] resultData = Binary.apply(a.flat, b.flat, operation);
    return new Tensor(resultData, a.dims);
  }

  /**
   * Adds two tensors elementwise.
   *
   * @param a the first tensor
   * @param b the second tensor
   * @return a new tensor containing the result of the addition
   */
  public static Tensor add(Tensor a, Tensor b) {
    return combine(a, b, (x, y) -> x + y);
  }

  /**
   * Multiplies two tensors elementwise (Hadamard product).
   *
   * @param a the first tensor
   * @param b the second tensor
   * @return a new tensor containing the result of the multiplication
   */
  public static Tensor hadamard(Tensor a, Tensor b) {
    return combine(a, b, (x, y) -> x * y);
  }

  /**
   * Exponentiates two tensors elementwise.
   *
   * @param a the base tensor
   * @param b the exponent tensor
   * @return a new tensor containing the result of the exponentiation
   */
  public static Tensor pow(Tensor a, Tensor b) {
    return combine(a, b, Math::pow);
  }

  public static Tensor contract(Tensor a, Tensor b, int[] indeciesA, int[] indicesB) {
    return null;
  }

  public static Tensor correlate(Tensor a, Tensor b) {
    return null;
  }

  public static Tensor outer(Tensor a, Tensor b) {
    return null;
  }

  public static Tensor transposedConvolution(Tensor a, Tensor b) {
    return null;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // INSTANCE TENSOR OPERATIONS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * Applies a unary operation to all elements of this tensor.
   *
   * @param operation the unary operation to apply
   * @return a new tensor containing the results
   */
  public Tensor apply(Unary operation) {return Tensor.apply(this, operation);}

  /**
   * Adds a constant value to all elements of this tensor.
   *
   * @param value the constant value to add
   * @return a new tensor containing the results
   */
  public Tensor add(double value) {return Tensor.add(this, value);}

  /**
   * Multiplies all elements of this tensor by a constant value.
   *
   * @param value the constant value to multiply by
   * @return a new tensor containing the results
   */
  public Tensor mul(double value) {return Tensor.mul(this, value);}

  /**
   * Combines this tensor with another tensor using a binary operation elementwise.
   *
   * @param other the other tensor to combine with
   * @param operation the binary operation to apply
   * @return a new tensor containing the results
   */
  public Tensor combine(Tensor other, Binary operation) {return Tensor.combine(this, other, operation);}

  /**
   * Adds another tensor to this tensor elementwise.
   *
   * @param other the other tensor to add
   * @return a new tensor containing the result of the addition
   */
  public Tensor add(Tensor other) {return Tensor.add(this, other);}

  /**
   * Multiplies this tensor with another tensor elementwise.
   *
   * @param other the other tensor to multiply
   * @return a new tensor containing the result of the multiplication
   */
  public Tensor hadamard(Tensor other) {return Tensor.hadamard(this, other);}

  /**
   * Exponentiates this tensor by another tensor elementwise.
   *
   * @param other the tensor containing the exponents
   * @return a new tensor containing the result of the exponentiation
   */
  public Tensor pow(Tensor other) {return Tensor.pow(this, other);}

  /**
   * Contracts this tensor with another tensor.
   *
   * @param b the other tensor to contract with
   * @param indeciesA the indices of the first tensor to contract along
   * @param indicesB the indices of the second tensor to contract along
   * @return a new tensor containing the result of the contraction
   */
  public Tensor contract(Tensor b, int[] indeciesA, int[] indicesB) {return Tensor.contract(this, b, indeciesA, indicesB);}

  /**
   * Correlates this tensor with another tensor.
   *
   * @param b the other tensor to correlate with
   * @return a new tensor containing the result of the correlation
   */
  public Tensor correlate(Tensor b) {return Tensor.correlate(this, b);}
  
  /**
   * Computes the outer product of two tensors
   * 
   * @param b the other tensor to compute the outer product with
   * @return a new tensor containing the result of the outer product
   */
  public Tensor outer(Tensor b) {return Tensor.outer(this, b);
  }

  /**
   * Computes the transposed convolution of two tensors
   * 
   * @param b the other tensor to compute the transposed convolution with
   * @return a new tensor containing the result of the transposed convolution
   */
  public Tensor transposedConvolution(Tensor b) {return Tensor.transposedConvolution(this, b);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // UTIL
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  public double get(int... indices) {
    return this.flat[Engine.getIndex(this.strides, indices)];
  }

  public int[] shape() {
    return dims.clone();
  }

  @Override
  public String toString() {
    return "Tensor" + Arrays.toString(dims) + " Data: " + Arrays.toString(this.flat);
  }
}
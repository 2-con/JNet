import java.util.Arrays;

import tensor.core.Engine;
import tensor.core.Generator;
import tensor.operation.Binary;
import tensor.operation.Unary;
import tensor.operation.Reduction;

public class Tensor {
  private final double[] data;
  public final int[] shape;
  public final int rank;
  public final Tensor T;
  public final int size;

  private final double[] grad;
  private final int[] strides;
  
  public Tensor(Object[] data, int... shape) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to shape " + Arrays.toString(shape));
    }

    double[] flat = new double[data.length];

    for (int i = 0; i < data.length; i++) {
      try {
        flat[i] = (double) data[i];
      } catch (Exception e) {
        throw new IllegalArgumentException("Unable to convert " + data[i] + " to double at index " + i);
      }
    }

    this.data = flat;
    this.grad = new double[data.length];
    this.shape = shape;
    this.strides = Engine.calculateStrides(shape);
    this.rank = shape.length;
    this.T = new Tensor(flat, Engine.reverse(shape), Engine.reverse(strides), this);
    this.size = flat.length;
  }

  public Tensor(double[] data, int... shape) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to shape " + Arrays.toString(shape));
    }

    this.data = data;
    this.grad = new double[data.length];
    this.shape = shape;
    this.strides = Engine.calculateStrides(shape);
    this.rank = shape.length;
    this.size = data.length;
    this.T = new Tensor(data, Engine.reverse(shape), Engine.reverse(strides), this);
  }

  private Tensor(double[] data, int[] strides, int[] shape) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to shape " + Arrays.toString(shape));
    }

    this.data = data;
    this.grad = new double[data.length];
    this.shape = shape;
    this.strides = strides;
    this.rank = shape.length;
    this.size = data.length;
    this.T = new Tensor(data, Engine.reverse(shape), Engine.reverse(strides), this);
  }

  private Tensor(double[] data, int[] shape, int[] strides, Tensor original) {
    this.data = data;
    this.shape = shape;
    this.strides = strides;
    this.rank = shape.length;
    this.grad = new double[data.length];
    this.T = original;
    this.size = data.length;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // TENSOR GENERATORS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * Returns a Tensor with random values uniformly distributed between 0 and 1.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and uniformly distributed values
   */
  public static Tensor randomUniform(int... shape) {return new Tensor(Generator.generateUniform(shape), shape);}

  /**
   * Returns a Tensor with random values drawn from a standard normal distribution.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and standard normally distributed values
   */
  public static Tensor randomNormal(int... shape) {return new Tensor(Generator.generateGaussian(shape), shape);}

  /**
   * Returns a Tensor with random values drawn from an exponential distribution.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and exponentially distributed values
   */
  public static Tensor randomExponential(int... shape) {return new Tensor(Generator.generateExponential(shape), shape);}

  /**
   * Returns a Tensor with all elements set to zero.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and all elements set to zero
   */
  public static Tensor zeros(int... shape) {return new Tensor(Generator.zeros(shape), shape);}

  /**
   * Returns a Tensor with all elements set to zero, with the same shapes as the given tensor.
   * 
   * @param tensor the tensor to return a zeros Tensor for
   * @return a new Tensor with the same shapes as the given tensor and all elements set to zero
   */
  public static Tensor zerosLike(Tensor tensor) {return zeros(tensor.shape);}

  /**
   * Returns a Tensor with all elements set to one.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and all elements set to one
   */
  public static Tensor ones(int... shape) {return new Tensor(Generator.ones(shape), shape);}

  /**
   * Returns a Tensor with all elements set to one, with the same shapes as the given tensor.
   * 
   * @param tensor the tensor to return a ones Tensor for
   * @return a new Tensor with the same shapes as the given tensor and all elements set to one
   */
  public static Tensor onesLike(Tensor tensor) {return ones(tensor.shape);}
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // STATIC TENSOR OPERATIONS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  // REDUCTION

  public static Tensor reduce(Tensor tensor, Reduction operation, int... axes) {
    return new Tensor(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, operation), Engine.getSurvivors(tensor.shape, axes));
  }

  public static Tensor sum(Tensor tensor, int... axes) {
    return new Tensor(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, Reduction.sum), Engine.getSurvivors(tensor.shape, axes));
  }

  public static Tensor prod(Tensor tensor, int... axes) {
    return new Tensor(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, Reduction.prod), Engine.getSurvivors(tensor.shape, axes));
  }

  public static Tensor min(Tensor tensor, int... axes) {
    return new Tensor(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, Reduction.min), Engine.getSurvivors(tensor.shape, axes));
  }

  public static Tensor max(Tensor tensor, int... axes) {
    return new Tensor(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, Reduction.max), Engine.getSurvivors(tensor.shape, axes));
  }

  public static Tensor mean(Tensor tensor, int... axes) {
    double[] data = Tensor.sum(tensor, axes).data;
    int divisor = 1;
    for (int axis : axes) divisor *= tensor.shape[axis];
    
    for (int i = 0; i < data.length; i++) data[i] /= divisor;
    return new Tensor(data, Engine.getSurvivors(tensor.shape, axes));
  }

  public static Tensor var(Tensor tensor, int... axes) {
    Tensor mu = Tensor.mean(tensor, axes);
    Tensor subtracted = Tensor.combine(tensor, mu, (a,b) -> a - b);
    Tensor squared = Tensor.pow(subtracted, 2);
    
    return Tensor.mul(squared, 1.0/tensor.size);
  }

  public static Tensor stdev(Tensor tensor, int... axes) {
    Tensor v = Tensor.var(tensor, axes);
    for (int i = 0; i < tensor.data.length; i++) tensor.data[i] = Math.sqrt(tensor.data[i]);
    return v;
  }

  // UNARY

  /**
   * Applies a unary operation to all elements of the given tensor.
   *
   * @param tensor the input tensor
   * @param operation the unary operation to apply
   * @return a new tensor containing the results
   */
  public static Tensor apply(Tensor tensor, Unary operation) {
    double[] resultData = Unary.apply(tensor.data, operation);
    return new Tensor(resultData, tensor.shape);
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

  /**
   * Computes the element-wise power of a tensor
   * 
   * @param tensor the input tensor
   * @param value the exponent
   * @return a new tensor containing the results
   */
  public static Tensor pow(Tensor tensor, double value) {
    return apply(tensor, x -> Math.pow(x, value));
  }

  public static Tensor permute(Tensor tensor, int... newOrder) {
    if (newOrder.length != tensor.shape.length) {
      throw new IllegalArgumentException("Permutation must include all axes.");
    }

    int[] newDims = new int[tensor.shape.length];
    int[] newStrides = new int[tensor.strides.length];

    for (int i = 0; i < newOrder.length; i++) {
      newDims[i] = tensor.shape[newOrder[i]];
      newStrides[i] = tensor.strides[newOrder[i]];
    }

    return new Tensor(tensor.data, newStrides, newDims);
  }

  public static Tensor transpose(Tensor tensor, int a, int b) {
    int[] order = new int[tensor.shape.length];
    for (int k = 0; k < tensor.shape.length; k++) order[k] = k;
    
    // Swap the two requested axes in the order array
    int temp = order[a];
    order[a] = order[b];
    order[b] = temp;
    
    return permute(tensor, order);
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
    if (!Arrays.equals(a.shape, b.shape)) {
      throw new RuntimeException("Incompatible shapes for Binary operation; got " + Arrays.toString(a.shape) + " and " + Arrays.toString(b.shape));
    }
    double[] resultData = Binary.apply(a.data, b.data, operation);
    return new Tensor(resultData, a.shape);
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

  /**
   * Contracts two tensors along the given axes.
   * 
   * The given axes in both tensors must have matching shapes.
   * 
   * @param a the first tensor
   * @param b the second tensor
   * @param axesA the axes of the first tensor to contract along
   * @param axesB the axes of the second tensor to contract along
   * @return a new tensor containing the result of the contraction
   * @throws IllegalArgumentException if the shapes of the two tensors are mismatched at the contraction axes
   */
  public static Tensor contract(Tensor a, Tensor b, int[] axesA, int[] axesB) {
    for (int i = 0; i < axesA.length; i++) {
      if (a.shape[axesA[i]] != b.shape[axesB[i]]) {
        throw new IllegalArgumentException("rankension mismatch at contraction axes.");
      }
    }

    int[] resShape = Engine.calculateResShape(a.shape, b.shape, axesA, axesB);
    double[] resultData = Engine.contract(a.data, a.strides, b.data, b.strides, a.shape, axesA, b.shape, axesB, resShape);

    return new Tensor(resultData, resShape);
  }

  public static Tensor correlate(Tensor a, Tensor b) {
    return null;
  }

  public static Tensor outer(Tensor a, Tensor b) {
    int[] emptyAxes = new int[0];
    
    int[] resShape = new int[a.shape.length + b.shape.length];
    System.arraycopy(a.shape, 0, resShape, 0, a.shape.length);
    System.arraycopy(b.shape, 0, resShape, a.shape.length, b.shape.length);

    double[] resData = Engine.contract(a.data, a.strides, b.data, b.strides, a.shape, emptyAxes, b.shape, emptyAxes, resShape);

    return new Tensor(resData, resShape);
  }

  public static Tensor transposedConvolution(Tensor a, Tensor b) {
    return null;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // INSTANCE TENSOR OPERATIONS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // REDUCTION

  /**
   * Apply a reduction operation to this tensor.
   *
   * @param operation the reduction operation to apply
   * @return a new tensor containing the results
   */
  public Tensor reduce(Reduction operation, int... axes) {return Tensor.reduce(this, operation, axes);}

  // UNARY

  /**
   * Apply a unary operation to all elements of this tensor.
   *
   * @param operation the unary operation to apply
   * @return a new tensor containing the results
   */
  public Tensor apply(Unary operation) {return Tensor.apply(this, operation);}

  /**
   * Add a constant value to all elements of this tensor.
   *
   * @param value the constant value to add
   * @return a new tensor containing the results
   */
  public Tensor add(double value) {return Tensor.add(this, value);}

  /**
   * Multiply all elements of this tensor by a constant value.
   *
   * @param value the constant value to multiply by
   * @return a new tensor containing the results
   */
  public Tensor mul(double value) {return Tensor.mul(this, value);}

  /**
   * Compute the element-wise power of this tensor
   * 
   * @param value the exponent
   * @return a new tensor containing the results
   */
  public Tensor pow(double value) {return Tensor.pow(this, value);}

  /**
   * Transpose the tensor according to a given permutation.
   *
   * @param permutation the permutation of axes to transpose the tensor
   * @return a new tensor containing the result of the transposition
   */
  public Tensor transpose(int a, int b) {return Tensor.transpose(this, a, b);}

  /**
   * Permute the tensor according to a given permutation.
   * 
   * @param newOrder the permutation of axes to permute the tensor
   * @return a new tensor containing the result of the permutation
   */
  public Tensor permute(int... newOrder) {return Tensor.permute(this, newOrder);}

  // BINARY

  /**
   * Combine this tensor with another tensor using a binary operation elementwise.
   *
   * @param other the other tensor to combine with
   * @param operation the binary operation to apply
   * @return a new tensor containing the results
   */
  public Tensor combine(Tensor other, Binary operation) {return Tensor.combine(this, other, operation);}

  /**
   * Add another tensor to this tensor elementwise.
   *
   * @param other the other tensor to add
   * @return a new tensor containing the result of the addition
   */
  public Tensor add(Tensor other) {return Tensor.add(this, other);}

  /**
   * Multiply this tensor with another tensor elementwise.
   *
   * @param other the other tensor to multiply
   * @return a new tensor containing the result of the multiplication
   */
  public Tensor hadamard(Tensor other) {return Tensor.hadamard(this, other);}

  /**
   * Raise this tensor by another tensor elementwise.
   *
   * @param other the tensor containing the exponents
   * @return a new tensor containing the result of the exponentiation
   */
  public Tensor pow(Tensor other) {return Tensor.pow(this, other);}

  /**
   * Contract this tensor with another tensor.
   *
   * @param b the other tensor to contract with
   * @param indeciesA the indices of the first tensor to contract along
   * @param indicesB the indices of the second tensor to contract along
   * @return a new tensor containing the result of the contraction
   */
  public Tensor contract(Tensor b, int[] indeciesA, int[] indicesB) {return Tensor.contract(this, b, indeciesA, indicesB);}

  /**
   * Cross-correlate this tensor with another kernel tensor.
   *
   * @param b the other tensor to correlate with
   * @return a new tensor containing the result of the correlation
   */
  public Tensor correlate(Tensor b) {return Tensor.correlate(this, b);}
  
  /**
   * Compute the outer product of this tensor and another tensor
   * 
   * @param b the other tensor to compute the outer product with
   * @return a new tensor containing the result of the outer product
   */
  public Tensor outer(Tensor b) {return Tensor.outer(this, b);}

  /**
   * Compute the transposed convolution of this tensor and another kernel tensor
   * 
   * @param b the other tensor to compute the transposed convolution with
   * @return a new tensor containing the result of the transposed convolution
   */
  public Tensor transposedConvolution(Tensor b) {return Tensor.transposedConvolution(this, b);}

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // UTIL
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * Returns the underlying flat array of this tensor.
   * 
   * @return the underlying flat array of this tensor
   */
  public double[] flatten() {
    return this.data.clone();
  }

  public double get(int... indices) {
    return this.data.clone()[Engine.getIndex(this.strides, indices)];
  }

  @Override
  public String toString() {
    if (this.data == null || this.shape == null || this.data.length == 0) return "[]";
    
    // Step 1: Find the maximum width for alignment
    int maxWidth = 0;
    for (double d : this.data) {
      maxWidth = Math.max(maxWidth, String.format("%.4f", d).length());
    }

    return "Tensor" + Arrays.toString(this.shape) + "\n" + Engine.prettyPrint(this.data, this.shape, this.strides, 0, 0, maxWidth);
  }
}
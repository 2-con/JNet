package tensor;
import java.util.Arrays;

import tensor.core.Engine;
import tensor.core.Generator;
import tensor.operation.Binary;
import tensor.operation.Unary;
import tensor.operation.Reduction;

public class RawTensor {
  private final double[] data;
  public final int[] shape;
  public final int rank;
  public final RawTensor T;
  public final int size;
  private final int[] strides;
  
  public RawTensor(Object[] data, int... shape) {
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
    this.shape = shape;
    this.strides = Engine.calculateStrides(shape);
    this.rank = shape.length;
    this.T = new RawTensor(flat, Engine.reverse(shape), Engine.reverse(strides), this);
    this.size = flat.length;
  }

  public RawTensor(double[] data, int... shape) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to shape " + Arrays.toString(shape));
    }

    this.data = data;
    this.shape = shape;
    this.strides = Engine.calculateStrides(shape);
    this.rank = shape.length;
    this.size = data.length;
    this.T = new RawTensor(data, Engine.reverse(shape), Engine.reverse(strides), this);
  }

  private RawTensor(double[] data, int[] strides, int[] shape) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to shape " + Arrays.toString(shape));
    }

    this.data = data;
    this.shape = shape;
    this.strides = strides;
    this.rank = shape.length;
    this.size = data.length;
    this.T = new RawTensor(data, Engine.reverse(shape), Engine.reverse(strides), this);
  }

  private RawTensor(double[] data, int[] shape, int[] strides, RawTensor original) {
    this.data = data;
    this.shape = shape;
    this.strides = strides;
    this.rank = shape.length;
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
  public static RawTensor randomUniform(int... shape) {return new RawTensor(Generator.generateUniform(shape), shape);}

  /**
   * Returns a Tensor with random values drawn from a standard normal distribution.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and standard normally distributed values
   */
  public static RawTensor randomNormal(int... shape) {return new RawTensor(Generator.generateGaussian(shape), shape);}

  /**
   * Returns a Tensor with random values drawn from an exponential distribution.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and exponentially distributed values
   */
  public static RawTensor randomExponential(int... shape) {return new RawTensor(Generator.generateExponential(shape), shape);}

  /**
   * Returns a Tensor with all elements set to zero.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and all elements set to zero
   */
  public static RawTensor zeros(int... shape) {return new RawTensor(Generator.zeros(shape), shape);}

  /**
   * Returns a Tensor with all elements set to zero, with the same shapes as the given tensor.
   * 
   * @param tensor the tensor to return a zeros Tensor for
   * @return a new Tensor with the same shapes as the given tensor and all elements set to zero
   */
  public static RawTensor zerosLike(RawTensor tensor) {return zeros(tensor.shape);}

  /**
   * Returns a Tensor with all elements set to one.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and all elements set to one
   */
  public static RawTensor ones(int... shape) {return new RawTensor(Generator.ones(shape), shape);}

  /**
   * Returns a Tensor with all elements set to one, with the same shapes as the given tensor.
   * 
   * @param tensor the tensor to return a ones Tensor for
   * @return a new Tensor with the same shapes as the given tensor and all elements set to one
   */
  public static RawTensor onesLike(RawTensor tensor) {return ones(tensor.shape);}
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // STATIC TENSOR OPERATIONS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  // REDUCTION

  public static RawTensor reduce(RawTensor tensor, Reduction operation, int... axes) {
    return new RawTensor(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, operation), Engine.getSurvivors(tensor.shape, axes));
  }

  // UNARY

  /**
   * Applies a unary operation to all elements of the given tensor.
   *
   * @param tensor the input tensor
   * @param operation the unary operation to apply
   * @return a new tensor containing the results
   */
  public static RawTensor apply(RawTensor tensor, Unary operation) {
    double[] resultData = Unary.apply(tensor.data, operation);
    return new RawTensor(resultData, tensor.shape);
  }

  public static RawTensor permute(RawTensor tensor, int... newOrder) {
    if (newOrder.length != tensor.shape.length) {
      throw new IllegalArgumentException("Permutation must include all axes.");
    }

    int[] newDims = new int[tensor.shape.length];
    int[] newStrides = new int[tensor.strides.length];

    for (int i = 0; i < newOrder.length; i++) {
      newDims[i] = tensor.shape[newOrder[i]];
      newStrides[i] = tensor.strides[newOrder[i]];
    }

    return new RawTensor(tensor.data, newStrides, newDims);
  }

  public static RawTensor transpose(RawTensor tensor, int a, int b) {
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
  public static RawTensor combine(RawTensor a, RawTensor b, Binary operation) {
    if (!Arrays.equals(a.shape, b.shape)) {
      throw new RuntimeException("Incompatible shapes for Binary operation; got " + Arrays.toString(a.shape) + " and " + Arrays.toString(b.shape));
    }
    double[] resultData = Binary.apply(a.data, b.data, operation);
    return new RawTensor(resultData, a.shape);
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
  public static RawTensor contract(RawTensor a, RawTensor b, int[] axesA, int[] axesB) {
    for (int i = 0; i < axesA.length; i++) {
      if (a.shape[axesA[i]] != b.shape[axesB[i]]) {
        throw new IllegalArgumentException("rankension mismatch at contraction axes.");
      }
    }

    int[] resShape = Engine.calculateResShape(a.shape, b.shape, axesA, axesB);
    double[] resultData = Engine.contract(a.data, a.strides, b.data, b.strides, a.shape, axesA, b.shape, axesB, resShape);

    return new RawTensor(resultData, resShape);
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
  public RawTensor reduce(Reduction operation, int... axes) {return RawTensor.reduce(this, operation, axes);}

  // UNARY

  /**
   * Apply a unary operation to all elements of this tensor.
   *
   * @param operation the unary operation to apply
   * @return a new tensor containing the results
   */
  public RawTensor apply(Unary operation) {return RawTensor.apply(this, operation);}

  /**
   * Transpose the tensor according to a given permutation.
   *
   * @param permutation the permutation of axes to transpose the tensor
   * @return a new tensor containing the result of the transposition
   */
  public RawTensor transpose(int a, int b) {return RawTensor.transpose(this, a, b);}

  /**
   * Permute the tensor according to a given permutation.
   * 
   * @param newOrder the permutation of axes to permute the tensor
   * @return a new tensor containing the result of the permutation
   */
  public RawTensor permute(int... newOrder) {return RawTensor.permute(this, newOrder);}

  // BINARY

  /**
   * Combine this tensor with another tensor using a binary operation elementwise.
   *
   * @param other the other tensor to combine with
   * @param operation the binary operation to apply
   * @return a new tensor containing the results
   */
  public RawTensor combine(RawTensor other, Binary operation) {return RawTensor.combine(this, other, operation);}

  /**
   * Contract this tensor with another tensor.
   *
   * @param b the other tensor to contract with
   * @param indeciesA the indices of the first tensor to contract along
   * @param indicesB the indices of the second tensor to contract along
   * @return a new tensor containing the result of the contraction
   */
  public RawTensor contract(RawTensor b, int[] indeciesA, int[] indicesB) {return RawTensor.contract(this, b, indeciesA, indicesB);}

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

    return "DataContainer" + Arrays.toString(this.shape) + "\n" + Engine.prettyPrint(this.data, this.shape, this.strides, 0, 0, maxWidth);
  }
}
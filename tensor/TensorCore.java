package tensor;
import java.util.Arrays;
import java.util.ArrayList;

import tensor.core.Engine;
import tensor.ops.Binary;
import tensor.ops.Reduction;
import tensor.ops.Unary;

public class TensorCore {
  private final double[] data;
  private final int[] shape;
  private final int rank;

  private final int size;
  private final int[] strides;
  
  public TensorCore(double[] data, int... shape) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to shape " + Arrays.toString(shape));
    }

    this.data = data.clone();
    this.shape = shape.clone();
    this.strides = Engine.calculateStrides(shape);
    this.rank = this.shape.length;
    this.size = this.data.length;
  }

  private TensorCore(double[] data, int[] shape, int[] strides) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be reshaped to shape " + Arrays.toString(shape));
    }

    this.data = data.clone();
    this.shape = shape.clone();
    this.strides = strides.clone(); // this constructor is private anyways, but just to make sure nothing silly happens
    this.rank = this.shape.length;
    this.size = this.data.length;
  }

  // ########################################################################################################### //
  //                                     STATIC TensorCore OPERATIONS                                            //
  // ########################################################################################################### //
  
  // SPCEIALS

  /**
   * Applies a permutation operation to the given tensor.
   * 
   * @param tensor the input tensor
   * @param newOrder the new order of the axes of the tensor
   * @return a new TensorCore containing the results
   * @throws IllegalArgumentException if the permutation axes are out of bounds, not unique, or lacks one or more axes
   */
  public static TensorCore permute(TensorCore tensor, int... newOrder) {
    ArrayList<Integer> indices = new ArrayList<>();

    for (int i = 0; i < newOrder.length; i++) {
      if (newOrder[i] < 0 || newOrder[i] >= tensor.shape.length) {
        throw new IllegalArgumentException("Permutation axes out of bounds");
      }
      
      for (Integer n : indices) {
        if (n == newOrder[i]) {
          throw new IllegalArgumentException("Permutation axes must be unique");
        }
      }

      indices.add(newOrder[i]);
    }

    if (newOrder.length != tensor.shape.length) {
      throw new IllegalArgumentException("Permutation must include all axes.");
    }

    int[] newDims = new int[tensor.shape.length];
    int[] newStrides = new int[tensor.strides.length];

    for (int i = 0; i < newOrder.length; i++) {
      newDims[i] = tensor.shape[newOrder[i]];
      newStrides[i] = tensor.strides[newOrder[i]];
    }

    return new TensorCore(tensor.data, newDims, newStrides);
  }

  // UNARY

  /**
   * Applies a reduction operation elements specified in the axes of the given tensor.
   *
   * @param TensorCore the input tensor
   * @param operation the reduction operation to apply
   * @param axes the axes to apply the reduction operation by
   * @return a new TensorCore containing the results
   */
  public static TensorCore reduce(TensorCore tensor, Reduction operation, int... axes) {
    return new TensorCore(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, operation), Engine.getSurvivors(tensor.shape, axes));
  }

  /**
   * Applies a unary operation to all elements of the given tensor.
   *
   * @param TensorCore the input tensor
   * @param operation the unary operation to apply
   * @return a new TensorCore containing the results
   */
  public static TensorCore apply(TensorCore tensor, Unary operation) {
    double[] resultData = Unary.apply(tensor.data, operation);
    return new TensorCore(resultData, tensor.shape);
  }

  // BINARY

  /**
   * Combines two tensors with a binary operation elementwise.
   *
   * @param a the first tensor
   * @param b the second tensor
   * @param operation the binary operation to apply
   * @return a new TensorCore containing the results
   * @throws RuntimeException if the shapes of the two tensors are mismatched
   */
  public static TensorCore combine(TensorCore a, TensorCore b, Binary operation) {
    if (!Arrays.equals(a.shape, b.shape)) {
      throw new RuntimeException("Incompatible shapes for Binary operation; got " + Arrays.toString(a.shape) + " and " + Arrays.toString(b.shape));
    }
    double[] resultData = Binary.apply(a.data, b.data, operation);
    return new TensorCore(resultData, a.shape);
  }

  /**
   * Contracts two tensors along the given axes.
   * 
   * The given axes in both tensors must have matching shapes.
   * 
   * @param a the first tensor
   * @param b the second tensor
   * @param axesA the axes of the first TensorCore to contract along
   * @param axesB the axes of the second TensorCore to contract along
   * @return a new TensorCore containing the result of the contraction
   * @throws IllegalArgumentException if the shapes of the two tensors are mismatched at the contraction axes
   */
  public static TensorCore contract(TensorCore a, TensorCore b, int[] axesA, int[] axesB) {
    for (int i = 0; i < axesA.length; i++) {
      if (a.shape[axesA[i]] != b.shape[axesB[i]]) {
        throw new IllegalArgumentException("rankension mismatch at contraction axes.");
      }
    }

    int[] resShape = Engine.calculateResShape(a.shape, b.shape, axesA, axesB);
    double[] resultData = Engine.contract(a.data, a.strides, b.data, b.strides, a.shape, axesA, b.shape, axesB, resShape);

    return new TensorCore(resultData, resShape);
  }
  
  // ########################################################################################################### //
  //                                     INSTANCE TensorCore OPERATIONS                                          //
  // ########################################################################################################### //

  // SPECIALS

  /**
   * Permute the TensorCore according to a given permutation.
   * 
   * @param newOrder the permutation of axes to permute the tensor
   * @return a new TensorCore containing the result of the permutation
   */
  public TensorCore permute(int... newOrder) {return TensorCore.permute(this, newOrder);}

  // UNARY

  /**
   * Apply a reduction operation to this tensor.
   *
   * @param operation the reduction operation to apply
   * @return a new TensorCore containing the results
   */
  public TensorCore reduce(Reduction operation, int... axes) {return TensorCore.reduce(this, operation, axes);}

  /**
   * Apply a unary operation to all elements of this tensor.
   *
   * @param operation the unary operation to apply
   * @return a new TensorCore containing the results
   */
  public TensorCore apply(Unary operation) {return TensorCore.apply(this, operation);}

  // BINARY

  /**
   * Combine this TensorCore with another TensorCore using a binary operation elementwise.
   *
   * @param other the other TensorCore to combine with
   * @param operation the binary operation to apply
   * @return a new TensorCore containing the results
   */
  public TensorCore combine(TensorCore other, Binary operation) {return TensorCore.combine(this, other, operation);}

  /**
   * Contract this TensorCore with another tensor.
   *
   * @param b the other TensorCore to contract with
   * @param indicesA the indices of the first TensorCore to contract along
   * @param indicesB the indices of the second TensorCore to contract along
   * @return a new TensorCore containing the result of the contraction
   */
  public TensorCore contract(TensorCore b, int[] indicesA, int[] indicesB) {return TensorCore.contract(this, b, indicesA, indicesB);}

  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  public int[] getShape() {return this.shape.clone();}

  public int getRank() {return this.rank;}

  public int getSize() {return this.size;}

  public int[] getStrides() {return this.strides.clone();}

  /**
   * Returns the raw memory buffer of this tensor.
   * 
   * @return the underlying flat array of this tensor
   */
  public double[] flatten() {
    return this.data.clone();
  }

  public double get(int... indices) {
    if (indices.length != this.rank) {
      throw new IllegalArgumentException("Invalid number of indices.");
    }

    return this.data[Engine.getIndex(this.strides, indices)];
  }

  @Override
  public String toString() {
    return "TensorCore" + prettyPrint();
  }

  public String prettyPrint() {
    if (this.data == null || this.shape == null || this.data.length == 0) return "[]";
    
    int maxWidth = 0;
    for (double d : this.data) {
      maxWidth = Math.max(maxWidth, String.format("%.4f", d).length());
    }

    return Arrays.toString(this.shape) + "\n" + Engine.prettyPrint(this.data, this.shape, this.strides, 0, 0, maxWidth);
  }
}
package tensor;
import java.util.ArrayList;
import java.util.Arrays;
import tensor.core.Engine;
import tensor.core.Memory;
import tensor.core.Utility;
import tensor.ops.Binary;
import tensor.ops.Reduction;
import tensor.ops.Unary;
import tensor.tools.ArrayTools;
import tensor.tools.Statistics;

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
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be shaped to " + Arrays.toString(shape));
    }

    this.data = data.clone();
    this.shape = shape.clone();
    this.strides = Memory.calculateStrides(shape);
    this.rank = this.shape.length;
    this.size = this.data.length;
  }

  private TensorCore(double[] data, int[] shape, int[] strides) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be shaped to " + Arrays.toString(shape));
    }

    this.data = data.clone();
    this.shape = shape.clone();
    this.strides = strides.clone(); // this constructor is private anyways, but just to make sure nothing silly happens
    this.rank = this.shape.length;
    this.size = this.data.length;
  }

  // ########################################################################################################### //
  //                                         TensorCore OPERATIONS                                               //
  // ########################################################################################################### //
  
  // SPECIAL

  /**
   * Permute the TensorCore according to a given permutation.
   * 
   * @param newOrder the permutation of axes to permute the tensor
   * @return a new TensorCore containing the result of the permutation
   */
  public TensorCore permute(int... newOrder) {return TensorCore.permute(this, newOrder);}
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

  /**
   * Broadcasts the TensorCore to have the specified axes.
   * 
   * @param newAxes the new axes to broadcast to
   * @return a new TensorCore containing the result of the broadcast
   */
  public TensorCore broadcast(int... targetShape) {return TensorCore.broadcast(this, targetShape);}
  /**
   * Broadcasts a tensor from one shape to another.
   * 
   * The dimensions of the input tensor and the target shape must match or one must be 1.
   * If the dimensions of the input tensor and the target shape are identical, the tensor is returned unchanged.
   * 
   * @param tensor the input tensor
   * @param targetShape the target shape of the tensor
   * @return a new TensorCore containing the broadcasted tensor
   * @throws RuntimeException if the dimensions of the input tensor and the target shape do not match or if one does not equal 1
   */
  public static TensorCore broadcast(TensorCore tensor, int... targetShape) {
    if (java.util.Arrays.equals(tensor.shape, targetShape)) {
      return tensor;
    }

    // Dimensions must match OR one must be 1
    if (tensor.shape.length != targetShape.length) {
      throw new RuntimeException("Mismatching rank for TensorCore broadcasting, attempting to broadcast " + java.util.Arrays.toString(tensor.shape) + " to " + java.util.Arrays.toString(targetShape));
    }

    for (int i = 0; i < tensor.shape.length; i++) {
      if (tensor.shape[i] != targetShape[i] && tensor.shape[i] != 1) {
        throw new RuntimeException("Mismatching dimension at axis " + i + ", attempting to broadcast " + tensor.shape[i] + " to " + targetShape[i]);
      }
    }

    double[] broadcastedData = Engine.broadcast(tensor.data, tensor.shape, targetShape);
    return new TensorCore(broadcastedData, targetShape);
  }

  /**
   * Squeezes the TensorCore to remove axes with size 1.
   *
   * @param axes the axes to squeeze
   * @return a new TensorCore containing the result of the squeeze operation
   */
  public TensorCore squeeze() {return TensorCore.squeeze(this);}
  /**
   * Squeeze the given tensor; all dimensions of size 1 are removed from the tensor
   * 
   * @param tensor the input tensor
   * @return a new TensorCore containing the input tensor with all dimensions of size 1 removed
   */
  public static TensorCore squeeze(TensorCore tensor) {
    return new TensorCore(tensor.data, Utility.squeezeShape(tensor.shape));
  }

  /**
   * Returns a new TensorCore containing the result of unsqueezing the input tensor along the specified axes.
   *
   * @param axes the axes to unsqueeze
   * @return a new TensorCore containing the result of the unsqueeze operation
   */
  public TensorCore unsqueeze(int... axes) {return TensorCore.unsqueeze(this, axes);}
  /**
   * Unsqueeze the given tensor along the specified axes; extra dimensions in the tensor are added in the specified indices.
   * 
   * @param tensor the input tensor
   * @param axes the axes to unqueeze the tensor along
   * @return a new TensorCore containing the input tensor with the specified axes expanded to size 1
   * @throws IllegalArgumentException if the specified axes are out of bounds of the input tensor
   */
  public static TensorCore unsqueeze(TensorCore tensor, int... axes) {
    for (int axis : axes) {
      if (axis < 0 || axis >= tensor.shape.length) {
        throw new IllegalArgumentException("Unsqueeze axes out of bounds: attempted to expand dimensions along axis " + axis);
      }
    }

    return new TensorCore(tensor.data, Utility.unsqueezeShape(tensor.shape, axes));
  }

  /**
   * Reshapes the given tensor into the specified shape.
   *
   * @param shape the desired shape of the output tensor
   * @return a new TensorCore containing the reshaped tensor
   * @throws IllegalArgumentException if the size of the input tensor does not match the size of the specified shape
   */
  public TensorCore reshape(int... shape) {return TensorCore.reshape(this, shape);}
  /**
   * Reshapes the given tensor into the specified shape.
   *
   * @param tensor the input tensor
   * @param shape the desired shape of the output tensor
   * @return a new TensorCore containing the reshaped tensor
   * @throws IllegalArgumentException if the size of the input tensor does not match the size of the specified shape
   */
  public static TensorCore reshape(TensorCore tensor, int... shape) {
    if (tensor.size != Statistics.prod(shape)) {
      throw new IllegalArgumentException("Insufficient elements to reshape tensor of shape " + Arrays.toString(tensor.shape) + " (size " + tensor.size + ") into a tensor of shape " + Arrays.toString(shape) + " (size " + tensor.size + ")");
    }

    int countNegatives = 0;
    for (int n : shape) {
      if (n < -1) {
        throw new IllegalArgumentException("Dimension inference only works on -1, got " + n + " instead");
      }
      if (n == -1) {
        countNegatives++;
      }
      if (countNegatives > 1) {
        throw new IllegalArgumentException("Only one dimension can be inferred, got " + countNegatives + " dimensions to infer");
      }
    }

    int[] newShape = Utility.inferShape(tensor.shape, shape);

    return new TensorCore(tensor.data, newShape);
  }

   // TODO: actually impliment this
  public static TensorCore concat(int axis, TensorCore... tensors) {
    System.out.println("Not implemented yet");
    return null;
  }

  // TODO: actually impliment this
  public static TensorCore stack(int axis, TensorCore... tensors) {
    System.out.println("Not implemented yet");
    return null;
  }

  // UNARY

  /**
   * Applies a reduction operation elements specified in the axes of the given tensor. reduce() defaults to preserving the dimension(s).
   *
   * @param TensorCore the input tensor
   * @param operation the reduction operation to apply
   * @param axes the axes to apply the reduction operation by
   * @return a new TensorCore containing the results
   */
  public TensorCore reduce(Reduction operation, int... axes) {return TensorCore.reduce(this, operation, axes);}
  /**
   * Applies a reduction operation elements specified in the axes of the given tensor. reduce() defaults to preserving the dimension(s).
   *
   * @param TensorCore the input tensor
   * @param operation the reduction operation to apply
   * @param axes the axes to apply the reduction operation by
   * @return a new TensorCore containing the results
   */
  public static TensorCore reduce(TensorCore tensor, Reduction operation, int... axes) {
    return new TensorCore(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, operation, true), Utility.getSurvivors(tensor.shape, axes));
  }

  /**
   * Apply a unary operation to all elements of this tensor.
   *
   * @param operation the unary operation to apply
   * @return a new TensorCore containing the results
   */
  public TensorCore apply(Unary operation) {return TensorCore.apply(this, operation);}
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
   * Combine this TensorCore with another TensorCore using a binary operation elementwise.
   *
   * @param other the other TensorCore to combine with
   * @param operation the binary operation to apply
   * @return a new TensorCore containing the results
   */
  public TensorCore elementwise(TensorCore other, Binary operation) {return TensorCore.elementwise(this, other, operation);}
  /**
   * Combines two tensors with a binary operation elementwise.
   *
   * @param a the first tensor
   * @param b the second tensor
   * @param operation the binary operation to apply
   * @return a new TensorCore containing the results
   * @throws RuntimeException if the shapes of the two tensors are mismatched
   */
  public static TensorCore elementwise(TensorCore a, TensorCore b, Binary operation) {
    if (a.rank != b.rank) {
      throw new IllegalArgumentException("Mismatching rank for binary elementwise operation. Got tensors of rank " + a.rank + " and " + b.rank);
    }
    if (!Arrays.equals(a.shape, b.shape)) {
      int[] broadcastShapeTarget = Utility.broadcastedShape(a.shape, b.shape);

      a = a.broadcast(broadcastShapeTarget);
      b = b.broadcast(broadcastShapeTarget);
    }

    double[] resultData = Binary.apply(a.data, b.data, operation);
    return new TensorCore(resultData, a.shape);
  }

  /**
   * Contract this TensorCore with another tensor.
   *
   * @param b the other TensorCore to contract with
   * @param indicesA the indices of the first TensorCore to contract along
   * @param indicesB the indices of the second TensorCore to contract along
   * @return a new TensorCore containing the result of the contraction
   */
  public TensorCore contract(TensorCore b, int[] indicesA, int[] indicesB) {return TensorCore.contract(this, b, indicesA, indicesB);}
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
    // if (axesA.length > a.rank) throw new IllegalArgumentException("Mismatching axes to contract Tensor A by, attempting to contract along " + axesA.length + " axes when Tensor A has a rank of " + a.rank );
    // if (axesB.length > b.rank) throw new IllegalArgumentException("Mismatching axes to contract Tensor B by, attempting to contract along " + axesB.length + " axes when Tensor A has a rank of " + b.rank);

    for (int i = 0; i < axesA.length; i++) {
      if (a.shape[axesA[i]] != b.shape[axesB[i]]) {
        throw new IllegalArgumentException("Mismatching dimensions at contraction axes, tensor A has size " + a.shape[axesA[i]] + " at index " + axesA[i] +  " while tensor B has size " + b.shape[axesB[i]]  + " at index " + axesB[i]);
      }
    }

    int[] resShape = Utility.calculateResShape(a.shape, b.shape, axesA, axesB);
    double[] resultData = Engine.contract(a.data, a.strides, b.data, b.strides, a.shape, axesA, b.shape, axesB, resShape);

    return new TensorCore(resultData, resShape);
  }
  
  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  public int[] getShape() {return this.shape.clone();}

  public int getRank() {return this.rank;}

  public int getSize() {return this.size;}

  public int[] getStrides() {return this.strides.clone();}

  /**
   * Returns a deep copy of the raw memory buffer of this tensor.
   * 
   * @return a deep copy of the underlying flat array of this tensor
   */
  public double[] dump() {
    return this.data.clone();
  }

  /**
   * Returns the raw memory buffer of this tensor. Unlike .dump(), this does not make a copy and is susceptible to data corruption if managed poorly!
   * 
   * @return the underlying flat array of this tensor
   */
  public double[] rawData() {
    return this.data; // unsafe, but whatever
  }

  public double get(int... indices) {
    if (indices.length != this.rank) {
      throw new IllegalArgumentException("Invalid number of indices.");
    }

    return this.data[Memory.getIndex(this.strides, indices)];
  }

  @Override
  public String toString() {
    if (this.data == null || this.shape == null || this.data.length == 0) return "TensorCore[null]";
    
    String prefix = "TensorCore" + Arrays.toString(this.shape) + "(\n";
    String content = ArrayTools.print(this.data, this.shape, this.strides, 0, 0, 2);

    return prefix + content + "\n)";
  }
}
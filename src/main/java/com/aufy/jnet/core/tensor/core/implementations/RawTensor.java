package com.aufy.jnet.core.tensor.core.implementations;
import java.util.Arrays;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.exceptions.tensors.Argument;
import com.aufy.jnet.core.backend.exceptions.tensors.Axes;
import com.aufy.jnet.core.backend.exceptions.tensors.Geometry;
import com.aufy.jnet.core.backend.exceptions.tensors.Operation;
import com.aufy.jnet.core.backend.interfaces.BinaryOp;
import com.aufy.jnet.core.backend.interfaces.UnaryOp;
import com.aufy.jnet.core.backend.scalarops.Miscellaneous;
import com.aufy.jnet.core.tensor.core.backend.compute.Memory;
import com.aufy.jnet.core.tensor.core.backend.interfaces.Reduction;
import com.aufy.jnet.core.tensor.core.backend.util.ArrayTools;
import com.aufy.jnet.core.tensor.functional.init.RawTensorGenerator;
import com.aufy.jnet.core.tensor.functional.main.RawBinaryOps;
import com.aufy.jnet.core.tensor.functional.main.RawReductionOps;
import com.aufy.jnet.core.tensor.functional.main.RawShapeOps;
import com.aufy.jnet.core.tensor.functional.main.RawUnaryOps;

public class RawTensor {

  protected double[] data;
  private int[] strides;
  private int[] shape;
  private int rank;
  private int size;
  
  public RawTensor(double[] data, int... shape) {
    this.data = data.clone();
    this.shape = shape.clone();
    this.strides = Memory.calculateStrides(shape);
    this.rank = this.shape.length;
    this.size = this.data.length;
  }

  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  public RawTensor duplicate() {return duplicate(this);}
  public static RawTensor duplicate(RawTensor tensor) {
    RawTensor out = new RawTensor(tensor.dump(), tensor.shape);
    
    // copy RawTensor attribute
    out.strides = tensor.strides.clone();

    return out;
  }

  public double[] dump() {return this.data.clone();}
  public int[] getStrides() {return this.strides.clone();}
  public int[] getShape() {return this.shape.clone();}
  public int getRank() {return this.rank;}
  public int getSize() {return this.size;}

  public double get(int... indices) {
    return this.data[Memory.getIndex(this.strides, indices)];
  }

  @Override
  public String toString() {
    if (this.data == null || this.shape == null || this.data.length == 0) return this.getClass().getSimpleName() + "[null]";
    
    String prefix = this.getClass().getSimpleName() + Arrays.toString(this.shape) + "(\n";
    String content = ArrayTools.print(this.data, this.shape, this.strides, 0, 0, 2);

    return prefix + content + "\n)";
  }

  // ########################################################################################################### //
  //                                              INITIALIZATION                                                 //
  // ########################################################################################################### //
  
  /**
   * Creates a tensor filled with ones.
   * 
   * @param shape the shape of the tensor.
   * @return the tensor filled with ones.
   */
  public static RawTensor ones(int... shape) {
    return RawTensorGenerator.ones(shape);
  }

  /**
   * Creates a tensor filled with ones according to the shape of this tensor. This tensor will not be modified.
   * 
   * @return the tensor filled with ones.
   */
  public RawTensor onesLike() {
    return ones(this.shape);
  }

  /**
   * Creates a tensor filled with ones according to the shape of another tensor.
   * 
   * @param tensor the tensor.
   * @return the tensor filled with ones.
   */
  public static RawTensor onesLike(RawTensor tensor) {
    return ones(tensor.shape);
  }

  /**
   * Creates a tensor filled with ones.
   * 
   * @param shape the shape of the tensor.
   * @return the tensor filled with ones.
   */
  public static RawTensor zeros(int... shape) {
    return RawTensorGenerator.zeros(shape);
  }

  /**
   * Creates a tensor filled with zeros according to the shape of this tensor. This tensor will not be modified.
   * 
   * @return the tensor filled with zeros.
   */
  public RawTensor zerosLike() {
    return zeros(this.shape);
  }

  /**
   * Creates a tensor filled with zeros according to the shape of another tensor.
   * 
   * @param tensor the tensor.
   * @return the tensor filled with zeros.
   */
  public static RawTensor zerosLike(RawTensor tensor) {
    return zeros(tensor.shape);
  }
  
  // ###########################################################################################################
  // ATOMIC OPERATIONS //
  // ###########################################################################################################

  // BINARY OPERATIONS ---------------------------------

  /**
   * Performs an elementwise binary operation on this tensor and another tensor. By default, RawTensor
   *
   * @param tensorB the second tensor involved in the operation.
   * @param operation the binary function to apply to elements.
   * @return a new tensor containing the results of the elementwise operation.
   */
  public RawTensor elementwise(RawTensor tensorB, BinaryOp operation) {
    return RawBinaryOps.elementwise(this, tensorB, operation);
  }

  /**
   * Contracts this tensor with another tensor along the specified axes.
   *
   * @param tensorB the tensor to contract with.
   * @param axesA the axes of contraction for this tensor.
   * @param axesB the axes of contraction for the second tensor.
   * @return a new tensor resulting from the contraction.
   */
  public RawTensor contract(RawTensor tensorB, int[] axesA, int[] axesB) {
    Axes.verifyNotEmpty("tensor contraction", axesA);
    Axes.verifyNotEmpty("tensor contraction", axesB);
    Axes.verifyAxis("tensor contraction", this.rank, Reductions.max(axesA));
    Axes.verifyAxis("tensor contraction", tensorB.rank, Reductions.max(axesB));

    return RawBinaryOps.contract(this, tensorB, axesA, axesB);
  }

  /**
   * Calculates the outer product of this tensor with another tensor to expand dimensions.
   *
   * @param tensorB the tensor to contract with.
   * @return a new tensor resulting from the product.
   */
  public RawTensor outer(RawTensor tensorB) {
    return RawBinaryOps.contract(this, tensorB, new int[0], new int[0]);
  }

  // REDUCTION OPERATIONS ---------------------------------

  /**
   * Reduces this tensor along specified axes using a given reduction operation.
   *
   * @param operation the reduction function to apply
   * @param axes the dimensions to reduce across
   * @return a new reduced tensor
   */
  public RawTensor reduce(Reduction operation, int... axes) {
    return RawReductionOps.reduce(this, operation, axes);
  }

  // SHAPE OPERATIONS ---------------------------------

  /**
   * Broadcasts this tensor to a new shape. Only singleton dimensions can be expanded.
   * 
   * @param shape the target shape.
   * @return a new tensor matched to the target shape.
   */
  public RawTensor broadcast(int... shape) {
    return RawShapeOps.broadcast(this, shape);
  }

  /**
   * Reshapes this tensor into a new target shape.
   *
   * @param shape the desired dimensions for the new tensor
   * @return a new tensor matched to the target shape layout
   */
  public RawTensor reshape(int... shape) {
    return RawShapeOps.reshape(this, shape);
  }

  /**
   * Permutes the axes order of this tensor according to the specified layout.
   *
   * @param axes the target ordering index map for the dimensions
   * @return a new tensor with reordered axes
   */
  public RawTensor permute(int... axes) {
    Axes.verifyUniqueList("permutation", axes);
    return RawShapeOps.permute(this, axes);
  }

  /**
   * Squeezes single-dimensional configurations out of this tensor's dimensions.
   *
   * @return a new tensor lacking single dimensions
   */
  public RawTensor squeeze() {
    return RawShapeOps.squeeze(this);
  }

  /**
   * Unsqueezes a single dimension into a specified tensor's structural layout.
   *
   * @param axes the location of the new dimensions to expand.
   * @return a new tensor expanded by an isolated dimension.
   */
  public RawTensor unsqueeze(int... axes) {
    Axes.verifyUniqueList("unsqueeze", axes);

    return RawShapeOps.unsqueeze(this, axes);
  }

  /**
   * Extracts a specific dimensional slice from this tensor layout.
   *
   * @param axis the dimension line targeted for slicing.
   * @param index the exact index position inside the targeted axis slice.
   * @return a new tensor representing the extracted slice.
   */
  public RawTensor slice(int axis, int index) {
    return RawShapeOps.slice(this, axis, index);
  }

  /**
   * Stacks this tensor with an array of source tensors uniformly together along a targeted axis.
   *
   * @param axis the axis line sequence configuration where tensors stack together.
   * @param tensors the target group of tensors being joined with this tensor.
   * @return a new stacked tensor representation.
   */
  public RawTensor stack(int axis, RawTensor... tensors) {
    RawTensor[] all = new RawTensor[tensors.length + 1];
    all[0] = this;
    System.arraycopy(tensors, 0, all, 1, tensors.length);
    return RawShapeOps.stack(axis, all);
  }

  /**
   * Concatenates this tensor with an array of source tensors together sequentially along a targeted axis.
   *
   * @param axis the operational axis sequence where arrays join up.
   * @param tensors the target group of tensors being concatenated with this tensor.
   * @return a new unified continuous structural tensor.
   */
  public RawTensor concat(int axis, RawTensor... tensors) {
    RawTensor[] all = new RawTensor[tensors.length + 1];
    all[0] = this;
    System.arraycopy(tensors, 0, all, 1, tensors.length);
    return RawShapeOps.concat(axis, all);
  }

  // UNARY OPERATIONS ---------------------------------

  /**
   * Applies an isolated unary operation function across elements of this tensor structure.
   *
   * @param operation the target operator function transformation sequence.
   * @param derivative the mathematical derivative algorithm representation for backwards sweeps.
   * @return a new transformed tensor layout sequence.
   */
  public RawTensor elementwise(UnaryOp operation) {
    return RawUnaryOps.elementwise(this, operation);
  }

  // ###########################################################################################################
  // SPECIFIC OPERATIONS //
  // ###########################################################################################################

  // BINARY OPERATIONS ---------------------------------

  /**
   * Adds another tensor to this tensor elementwise.
   *
   * @param tensorB the tensor to add.
   * @return a new tensor containing the elementwise sum.
   */
  public RawTensor add(RawTensor tensorB) {
    return elementwise(tensorB, (x, y) -> x + y);
  }

  /**
   * Subtracts another tensor from this tensor elementwise.
   *
   * @param tensorB the tensor to subtract.
   * @return a new tensor containing the elementwise difference.
   */
  public RawTensor sub(RawTensor tensorB) {
    return elementwise(tensorB, (x, y) -> x - y);
  }

  /**
   * Computes the Hadamard product (elementwise multiplication) between this tensor and another.
   *
   * @param tensorB the tensor to multiply by.
   * @return a new tensor containing the elementwise product.
   */
  public RawTensor hadamard(RawTensor tensorB) {
    return elementwise(tensorB, (x, y) -> x * y);
  }

  /**
   * Divides this tensor by another tensor elementwise.
   *
   * @param tensorB the denominator tensor.
   * @return a new tensor containing the elementwise quotient.
   */
  public RawTensor div(RawTensor tensorB) {
    return elementwise(tensorB, (x, y) -> x / y);
  }

  /**
   * Raises this tensor to the power of another tensor elementwise.
   *
   * @param tensorB the exponent tensor.
   * @return a new tensor containing the elementwise power results.
   */
  public RawTensor pow(RawTensor tensorB) {
    return elementwise(tensorB, (x, y) -> Math.pow(x, y));
  }

  /**
   * Multiplies this tensor by another tensor using matrix multiplication.
   *
   * @param tensorB the multiplier tensor.
   * @return a new tensor containing the matrix product.
   */
  public RawTensor matmul(RawTensor tensorB) {
    Operation.verifyMatMul(this.shape, tensorB.shape);

    int[] axesA = {this.rank - 1};
    int[] axesB = {(tensorB.rank == 1) ? 0 : tensorB.rank - 2 };

    return contract(tensorB, axesA, axesB);
  }

  // REDUCTION OPERATIONS ---------------------------------

  /**
   * Sums the elements of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the sums.
   */
  public RawTensor sum(int... axes) {
    return RawReductionOps.reduce(this, Reductions::sum, axes);
  }

  /**
   * Computes the product of the elements of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the products.
   */
  public RawTensor prod(int... axes) {
    return RawReductionOps.reduce(this, Reductions::prod, axes);
  }

  /**
   * Finds the minimum values of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the minimums.
   */
  public RawTensor min(int... axes) {
    return RawReductionOps.reduce(this, Reductions::min, axes);
  }

  /**
   * Finds the maximum values of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the maximums.
   */
  public RawTensor max(int... axes) {
    return RawReductionOps.reduce(this, Reductions::max, axes);
  }

  // SHAPE OPERATIONS ---------------------------------

  /**
   * Flattens this tensor into a single-dimensional layout.
   *
   * @return a new flattened tensor.
   */
  public RawTensor flatten() {
    return reshape(-1);
  }

  /**
   * Transposes the dimensional axes of this tensor.
   *
   * @return a new transposed tensor.
   */
  public RawTensor transpose() {
    Geometry.verifyMinimumRank("transpose", 2, this.rank);

    int[] transposed = Tools.arrange(this.rank);

    // swapping the last 2 axes
    int temp = transposed[this.rank - 1];
    transposed[this.rank - 1] = transposed[this.rank - 2];
    transposed[this.rank - 2] = temp;

    return permute(transposed);
  }

  // UNARY OPERATIONS ---------------------------------

  /**
   * Adds a scalar value to every element in this tensor.
   *
   * @param scalar the value to add.
   * @return a new shifted tensor.
   */
  public RawTensor add(double scalar) {
    return elementwise((a) -> a + scalar);
  }

  /**
   * Multiplies every element in this tensor by a scalar value.
   *
   * @param scalar the scaling factor.
   * @return a new scaled tensor.
   */
  public RawTensor mul(double scalar) {
    return elementwise((a) -> a * scalar);
  }

  /**
   * Raises every element in this tensor to a scalar power.
   *
   * @param scalar the exponent value.
   * @return a new exponentiated tensor.
   */
  public RawTensor pow(double scalar) {
    return elementwise((a) -> Math.pow(a, scalar));
  }

  
  /**
   * Exponentiates every element in this tensor to a scalar power (scalar^element).
   *
   * @param scalar the exponent value.
   * @return a new exponentiated tensor.
   */
  public RawTensor exp(double scalar) {
    Argument.aboveValue(0, scalar, "exp");
    return elementwise((a) -> Math.pow(scalar, a));
  }

  /**
   * Exponentiates every element in this tensor.
   *
   * @return a new exponentiated tensor.
   */
  public RawTensor exp() {
    return elementwise((a) -> Math.exp(a));
  }

  
  /**
   * Take the logarithm of every element in this tensor using a specified base.
   *
   * @return a new log tensor.
   */
  public RawTensor log(double base) {
    Operation.aboveValue(0, this.dump(), "log");
    Argument.aboveValue(0, base, "log");
    return elementwise((a) -> Math.log(a)/Math.log(base));
  }

  /**
   * Take the logarithm of every element in this tensor using a specified base.
   *
   * @return a new log tensor.
   */
  public RawTensor ln() {
    Operation.aboveValue(0, this.dump(), "log");
    return elementwise((a) -> Math.log(a));
  }

  /**
   * Returns the absolute value of each element in this tensor.
   *
   * @return a new tensor containing the absolute values.
   */
  public RawTensor abs() {
    return elementwise(Math::abs);
  }

  /**
   * Returns the sign of each element in this tensor.
   *
   * @return a new tensor containing the signs.
   */
  public RawTensor sign() {
    return elementwise(Math::signum);
  }

  /**
   * Returns the ceiling of each element in this tensor.
   *
   * @return a new tensor containing the ceilings.
   */
  public RawTensor ceil() {
    return elementwise(Math::ceil);
  }

  /**
   * Returns the floor of each element in this tensor.
   *
   * @return a new tensor containing the floors.
   */
  public RawTensor floor() {
    return elementwise(Math::floor);
  }

  /**
   * Returns the rounded value of each element in this tensor.
   *
   * @param scale the number of decimal places to round to.
   * @return a new tensor containing the rounded values.
   */
  public RawTensor round(int scale) {
    Argument.belowValue(0, scale, "rounding");
    return elementwise(a -> Miscellaneous.round(a, scale));
  }

  /**
   * Returns the truncated value of each element in this tensor.
   *
   * @param scale the number of decimal places to truncate to.
   * @return a new tensor containing the truncated values.
   */
  public RawTensor trunc(int scale) {
    Argument.belowValue(0, scale, "truncate");
    return elementwise(a -> Miscellaneous.truncate(a, scale));
  }

  /**
   * Returns the fractional part of each element in this tensor.
   *
   * @param scale the number of decimal places to keep for the fractional part.
   * @return a new tensor containing the fractional parts.
   */
  public RawTensor frac(int scale) {
    Argument.belowValue(0, scale, "fractional");
    return elementwise(a -> Miscellaneous.fractional(a, scale));
  }

  /**
   * Returns the clipped value of each element in this tensor clipped between a min and a max value.
   *
   * @param tensor the tensor to evaluate.
   * @param min minimum allowed value.
   * @param max maximum allowed value.
   * @return a new tensor containing the clipped values.
   */
  public RawTensor clip(double min, double max) {
    return elementwise(a -> Math.min(Math.max(a, min), max));
  }
}
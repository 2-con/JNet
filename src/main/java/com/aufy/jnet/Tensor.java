package com.aufy.jnet;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.exceptions.internal.Data;
import com.aufy.jnet.core.backend.exceptions.tensors.Argument;
import com.aufy.jnet.core.backend.exceptions.tensors.Axes;
import com.aufy.jnet.core.backend.exceptions.tensors.Geometry;
import com.aufy.jnet.core.backend.exceptions.tensors.Graphing;
import com.aufy.jnet.core.backend.exceptions.tensors.Operation;
import com.aufy.jnet.core.backend.interfaces.BinaryOp;
import com.aufy.jnet.core.backend.interfaces.UnaryOp;
import com.aufy.jnet.core.backend.scalarops.Miscellaneous;
import com.aufy.jnet.core.backend.scalarops.Unary;
import com.aufy.jnet.core.tensor.core.backend.interfaces.Reduction;
import com.aufy.jnet.core.tensor.core.backend.util.ArrayTools;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.graph.main.BinaryOps;
import com.aufy.jnet.core.tensor.graph.main.ReductionOps;
import com.aufy.jnet.core.tensor.graph.main.ShapeOps;
import com.aufy.jnet.core.tensor.graph.main.UnaryOps;
import com.aufy.jnet.statistics.Generator;
import com.aufy.jnet.statistics.distributions.Distribution;

/**
 * The main Tensor class of JNet. This class is a wrapper around the CoreTensor
 * class but it is advised to use the CoreTensor class directly because
 * there are no built in methods and there are no safety checks (unlike this
 * specialized class).
 */
public class Tensor {
  private final CoreTensor core;

  private final int[] shape;
  public final int rank;
  public final int size;

  public static boolean verbose = false;
  public boolean enableGrad;

  public Tensor(double[] data, int... shape) {
    Geometry.verifyNotEmpty("initialization", shape);
    Geometry.verifyDataShape("initialization", data.length, shape);

    this.shape = shape;
    this.rank = shape.length;
    this.size = data.length;
    this.enableGrad = false;

    this.core = new CoreTensor(data, shape);
  }

  public Tensor(Distribution distribution, int... shape) {
    Geometry.verifyNotEmpty("initialization", shape);

    double[] data = Generator.sample(distribution, shape);

    this.shape = shape;
    this.rank = shape.length;
    this.size = data.length;
    this.enableGrad = false;

    this.core = new CoreTensor(data, shape);
  }

  public Tensor(CoreTensor tensor) {
    this.shape = tensor.shape.clone();
    this.rank = tensor.rank;
    this.size = tensor.size;
    this.enableGrad = tensor.requiresGrad;

    this.core = tensor.duplicate();
  }

  // ###########################################################################################################
  // UTILITY //
  // ###########################################################################################################

  @Override
  public String toString() {
    if (this.core.dump() == null || this.shape == null || this.core.dump().length == 0)
      return "Tensor[null]";

    String prefix = "Tensor" + Arrays.toString(this.shape) + "(\n";
    String content = ArrayTools.print(this.core.dump(), this.shape, this.core.core.getStrides(), 0, 0, 2);
    String suffix = " grad=" + this.enableGrad;

    if (verbose) {
      return prefix + content + "\n\n" + suffix + "\n)";
    } else {
      return prefix + content + "\n)";
      // return prefix + "\n)";
    }
  }

  public Tensor duplicate() {
    return new Tensor(this.core.duplicate());
  }

  public static Tensor duplicate(Tensor tensor) {
    return new Tensor(tensor.core.duplicate());
  }

  public double[] dump() {
    return dump(this);
  }

  public static double[] dump(Tensor tensor) { // safe dump
    Data.verifyData(tensor.core.dump());
    return tensor.core.dump().clone();
  }

  public int[] getShape() {return this.shape.clone();}
  
  public int getShape(int index) {return this.shape[index];}

  // ###########################################################################################################
  // INITIALIZATION //
  // ###########################################################################################################

  /**
   * Creates a tensor filled with ones.
   * 
   * @param shape the shape of the tensor.
   * @return the tensor filled with ones.
   */
  public static Tensor ones(int... shape) {
    return new Tensor(CoreTensor.ones(shape));
  }

  /**
   * Creates a tensor filled with ones according to the shape of this tensor. This tensor will not be modified.
   * 
   * @return the tensor filled with ones.
   */
  public Tensor onesLike() {
    return ones(this.shape);
  }

  /**
   * Creates a tensor filled with ones according to the shape of another tensor.
   * 
   * @param tensor the tensor.
   * @return the tensor filled with ones.
   */
  public static Tensor onesLike(Tensor tensor) {
    return ones(tensor.shape);
  }

  /**
   * Creates a tensor filled with ones.
   * 
   * @param shape the shape of the tensor.
   * @return the tensor filled with ones.
   */
  public static Tensor zeros(int... shape) {
    return new Tensor(CoreTensor.zeros(shape));
  }

  /**
   * Creates a tensor filled with zeros according to the shape of this tensor. This tensor will not be modified.
   * 
   * @return the tensor filled with zeros.
   */
  public Tensor zerosLike() {
    return zeros(this.shape);
  }

  /**
   * Creates a tensor filled with zeros according to the shape of another tensor.
   * 
   * @param tensor the tensor.
   * @return the tensor filled with zeros.
   */
  public static Tensor zerosLike(Tensor tensor) {
    return zeros(tensor.shape);
  }

  // ###########################################################################################################
  // AUTOGRAD //
  // ###########################################################################################################

  /**
   * Computes the gradients of this tensor with respect to the graph leaves.
   * Runs the backward pass starting from this tensor.
   *
   * @throws IllegalStateException if the tensor is not configured to track gradients.
   */
  public void backward() {
    Graphing.verifyDifferentiable("backwards", core.requiresGrad);
    core.backward();
  }

  /**
   * Computes the gradients of a tensor with respect to the graph leaves.
   * Runs the backward pass starting from the specified tensor.
   *
   * @param tensor the tensor to run the backward pass from.
   * @throws IllegalStateException if the tensor is not configured to track gradients.
   */
  public static void backward(Tensor tensor) {
    Graphing.verifyDifferentiable("backwards", tensor.core.requiresGrad);
    tensor.core.backward();
  }

  /**
   * Retrieves the accumulated gradient tensor for this tensor instance.
   *
   * @return a new tensor containing the tracked gradient values.
   * @throws IllegalStateException if gradient tracking is not enabled.
   */
  public Tensor grad() {
    return grad(this);
  }

  /**
   * Retrieves the accumulated gradient tensor for the specified tensor.
   *
   * @param tensor the source tensor to extract the gradient from.
   * @return a new tensor containing the tracked gradient values.
   * @throws IllegalStateException if gradient tracking is not enabled.
   */
  public static Tensor grad(Tensor tensor) {
    Graphing.verifyDifferentiable("grads", tensor.core.requiresGrad);
    Operation.hasGrad(tensor.core.grad);
    return new Tensor(tensor.core.grad);
  }

  /**
   * Disables gradient tracking for this tensor instance.
   */
  public void noGrad() {
    core.noGrad();
  }

  /**
   * Disables gradient tracking for the specified core tensor.
   *
   * @param tensor the target core tensor structure to modify
   */
  public static void noGrad(CoreTensor tensor) {
    tensor.noGrad();
  }

  /**
   * Detaches this tensor from the current computational graph.
   * The returned tensor will not track further operations for autograd.
   *
   * @return a new isolated tensor detached from the graph history.
   */
  public Tensor detach() {
    return detach(this);
  }

  /**
   * Detaches a specified tensor from its computational graph.
   * The returned tensor will not track further operations for autograd.
   *
   * @param tensor the source tensor to isolate.
   * @return a new isolated tensor detached from the graph history.
   */
  public static Tensor detach(Tensor tensor) {
    return new Tensor(CoreTensor.detach(tensor.core));
  }

  /**
   * Resets the accumulated gradients of this tensor instance back to zero.
   */
  public void zeroGrad() {
    zeroGrad(this);
  }

  /**
   * Resets the accumulated gradients of the specified tensor back to zero.
   *
   * @param tensor the target tensor whose gradients should be cleared.
   */
  public static void zeroGrad(Tensor tensor) {
    tensor.core.zeroGrad();
  }

  /**
   * Enables autograd tracking flags for this tensor instance.
   * Makes this tensor and future operations eligible for backward graph passes.
   * 
   * @return this tensor with autograd configuration enabled.
   */
  public Tensor requiresGrad() {
    return requiresGrad(this);
  }

  /**
   * Enables autograd tracking flags for the specified tensor.
   * Makes the tensor and future operations eligible for backward graph passes.
   *
   * @param tensor the target tensor to configure.
   * @return the same tensor instance with autograd configuration enabled.
   */
  public static Tensor requiresGrad(Tensor tensor) {
    tensor.enableGrad = true;
    tensor.core.requiresGrad = true;
    return tensor;
  }

  // ###########################################################################################################
  // GENERAL OPERATIONS //
  // ###########################################################################################################

  // UNARY OPERATIONS ---------------------------------

  /**
   * Applies a unary operation to the specified tensor. Autograd ignores everything inside forwardOperation and uses derivative to propagate gradients.
   * Because gradients are not tracked, forwardOperation use the RawTensor tensor (that dosn't natively track gradients) as the main tensor type (not this Tensor class).
   * Syntax will not differ much from Tensor aside from minor differences in parameters of certian methods.
   * 
   * Tensor will not double-check gradient correctness.
   * 
   * @param forwardOperation forward operation applied to tensor (symbolically) as operation.apply(tensor).
   * @param derivative backward operation applied to the incoming gradient tensor as derivative.apply(grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @return this Tensor after the operation has been applied.
   */
  public Tensor apply(Function<RawTensor, RawTensor> forwardOperation, Function<RawTensor, RawTensor> derivative) {
    return new Tensor(UnaryOps.apply(this.core, forwardOperation, derivative)); 
  }

  /**
   * Applies a unary operation to the specified tensor. Autograd ignores everything inside forwardOperation and uses derivative to propagate gradients.
   * Because gradients are not tracked, forwardOperation use the RawTensor tensor (that dosn't natively track gradients) as the main tensor type (not this Tensor class).
   * Syntax will not differ much from Tensor aside from minor differences in parameters of certian methods.
   * 
   * Tensor will not double-check gradient correctness.
   * 
   * @param tensor the tensor to apply the operation to.
   * @param forwardOperation forward operation applied to tensor (symbolically) as operation.apply(tensor).
   * @param derivative backward operation applied to the incoming gradient tensor as derivative.apply(grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @return a Tensor after the operation has been applied.
   */
  public static Tensor apply(Tensor tensor, Function<RawTensor, RawTensor> forwardOperation, Function<RawTensor, RawTensor> derivative) {
    return new Tensor(UnaryOps.apply(tensor.core, forwardOperation, derivative)); 
  }

  // BINARY OPERATIONS ---------------------------------

  /**
   * Applies a binary operation to the specified tensors. Autograd ignores everything inside forwardOperation and uses dA and dB to propagate partial gradients with respect to each tensor respectively.
   * Because gradients are not tracked, forwardOperation use the RawTensor tensor (that dosn't natively track gradients) as the main tensor type (not this Tensor class).
   * Syntax will not differ much from Tensor aside from minor differences in parameters of certian methods.
   * 
   * Tensor will not double-check gradient correctness.
   * 
   * @param tensorB the second tensor to apply the operation to.
   * @param forwardOperation forward operation applied to tensor (symbolically) as operation.apply(tensorA, tensorB).
   * @param dA partial derivative function of the incoming gradient tensor with respect to the first tensor applied as dA.apply(tensorB, grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @param dB partial derivative function of the incoming gradient tensor with respect to the second tensor applied as dB.apply(tensorA, grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @return this Tensor after the operation has been applied.
   */
  public Tensor apply(Tensor tensorB, BiFunction<RawTensor, RawTensor, RawTensor> forwardOperation, BiFunction<RawTensor, RawTensor, RawTensor> dA, BiFunction<RawTensor, RawTensor, RawTensor> dB) {
    return new Tensor(BinaryOps.apply(this.core, tensorB.core, forwardOperation, dA, dB)); 
  }

  /**
   * Applies a binary operation to the specified tensors. Autograd ignores everything inside forwardOperation and uses dA and dB to propagate partial gradients with respect to each tensor respectively.
   * Because gradients are not tracked, forwardOperation use the RawTensor tensor (that dosn't natively track gradients) as the main tensor type (not this Tensor class).
   * Syntax will not differ much from Tensor aside from minor differences in parameters of certian methods.
   * 
   * Tensor will not double-check gradient correctness.
   * 
   * @param tensorA the first tensor to apply the operation to.
   * @param tensorB the second tensor to apply the operation to.
   * @param forwardOperation forward operation applied to tensor (symbolically) as operation.apply(tensorA, tensorB).
   * @param dA partial derivative function of the incoming gradient tensor with respect to the first tensor applied as dA.apply(tensorB, grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @param dB partial derivative function of the incoming gradient tensor with respect to the second tensor applied as dB.apply(tensorA, grad). This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @return a Tensor after the operation has been applied.
   */
  public static Tensor apply(Tensor tensorA, Tensor tensorB, BiFunction<RawTensor, RawTensor, RawTensor> forwardOperation, BiFunction<RawTensor, RawTensor, RawTensor> dA, BiFunction<RawTensor, RawTensor, RawTensor> dB) {
    return new Tensor(BinaryOps.apply(tensorA.core, tensorB.core, forwardOperation, dA, dB)); 
  }

  // ###########################################################################################################
  // ATOMIC OPERATIONS //
  // ###########################################################################################################

  // BINARY OPERATIONS ---------------------------------

  /**
   * Performs an elementwise binary operation on this tensor and another tensor.
   *
   * @param tensorB the second tensor involved in the operation.
   * @param forwardOperation the binary function to apply to elements.
   * @param dA the partial derivative function with respect to the first tensor.
   * @param dB the partial derivative function with respect to the second tensor.
   * @return a new tensor containing the results of the elementwise operation.
   */
  public Tensor elementwise(Tensor tensorB, BinaryOp forwardOperation, BinaryOp dA, BinaryOp dB) {
    return elementwise(this, tensorB, forwardOperation, dA, dB);
  }

  /**
   * Performs an elementwise binary operation between two specified tensors.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @param forwardOperation the binary function to apply to elements.
   * @param dA the partial derivative function with respect to tensorA.
   * @param dB the partial derivative function with respect to tensorB.
   * @return a new tensor containing the results of the elementwise operation.
   */
  public static Tensor elementwise(Tensor tensorA, Tensor tensorB, BinaryOp forwardOperation, BinaryOp dA, BinaryOp dB) {
    return new Tensor(BinaryOps.elementwise(tensorA.core, tensorB.core, forwardOperation, dA, dB));
  }

  // UNARY OPERATIONS ---------------------------------

  /**
   * Applies an isolated unary operation function across elements of this tensor structure.
   *
   * @param forwardOperation the target operator function transformation sequence.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @return a new transformed tensor layout sequence.
   */
  public Tensor elementwise(UnaryOp forwardOperation, UnaryOp derivative) {
    return elementwise(this, forwardOperation, derivative);
  }

  /**
   * Applies an isolated unary operation function across elements of a specified
   * tensor.
   *
   * @param tensor the tensor to modify.
   * @param forwardOperation the target operator function transformation sequence.
   * @param derivative the derivative function to find the local derivative, which is then multiplied by incoming gradients for the final result.
   * @return a new transformed tensor layout sequence.
   */
  public static Tensor elementwise(Tensor tensor, UnaryOp forwardOperation, UnaryOp derivative) {
    return new Tensor(UnaryOps.elementwise(tensor.core, forwardOperation, derivative));
  }

  // REDUCTION OPERATIONS ---------------------------------

  /**
   * Reduces this tensor along specified axes using a given reduction operation. Derivatives must be supplied to find the local derivative of the reduction function; Tensor
   * will not check for gradient correctness.
   *
   * @param operation the reduction function to apply.
   * @param derivative the derivative of the reduction function, applied to a broadcasted output tensor and the orginal tensor. This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @param axes the dimensions to reduce across.
   * @return a new reduced tensor.
   */
  public Tensor reduce(Reduction operation, BiFunction<RawTensor, RawTensor, RawTensor> derivative, int... axes) {
    return reduce(this, operation, derivative, axes);
  }

  /**
   * Reduces this tensor along specified axes using a given reduction operation. Derivatives must be supplied to find the local derivative of the reduction function; Tensor
   * will not check for gradient correctness.
   *
   * @param tensor the tensor to apply the operation to.
   * @param operation the reduction function to apply.
   * @param derivative the derivative of the reduction function, applied to a broadcasted output tensor and the orginal tensor. This function is to find the local derivative
   * as the result of this function will be multiplied by the incoming gradient tensor elementwise (hadamard product).
   * @param axes the dimensions to reduce across.
   * @return a new reduced tensor.
   */
  public static Tensor reduce(Tensor tensor, Reduction operation, BiFunction<RawTensor, RawTensor, RawTensor> derivative, int... axes) {
    return new Tensor(ReductionOps.reduce(tensor.core, operation, derivative, axes));
  }

  // ###########################################################################################################
  // SPECIFIC OPERATIONS //
  // ###########################################################################################################

  // BINARY OPERATIONS ---------------------------------

  /**
   * Contracts this tensor with another tensor along the specified axes.
   *
   * @param tensorB the tensor to contract with.
   * @param axesA the axes of contraction for this tensor.
   * @param axesB the axes of contraction for the second tensor.
   * @return a new tensor resulting from the contraction.
   */
  public Tensor contract(Tensor tensorB, int[] axesA, int[] axesB) {
    return contract(this, tensorB, axesA, axesB);
  }

  /**
   * Contracts two specified tensors along their respective axes locations. Tensor contraction is a generalization of matrix multiplication; resulting tensors are automatically squeezed to remove single-dimensional entries.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @param axesA the axes of contraction for tensorA.
   * @param axesB the axes of contraction for tensorB.
   * @return a new tensor resulting from the contraction.
   */
  public static Tensor contract(Tensor tensorA, Tensor tensorB, int[] axesA, int[] axesB) {
    Axes.verifyNotEmpty("tensor contraction", axesA);
    Axes.verifyNotEmpty("tensor contraction", axesB);
    Axes.verifyAxis("tensor contraction", tensorA.rank, Reductions.max(axesA));
    Axes.verifyAxis("tensor contraction", tensorB.rank, Reductions.max(axesB));

    return new Tensor(BinaryOps.contract(tensorA.core, tensorB.core, axesA, axesB));
  }

  /**
   * Calculates the outer product of this tensor with another tensor to expand dimensions.
   *
   * @param tensorB the tensor to contract with.
   * @return a new tensor resulting from the product.
   */
  public Tensor outer(Tensor tensorB) {
    return outer(this, tensorB);
  }

  /**
   * Calculates the outer product of this tensor with another tensor to expand dimensions.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor resulting from the product.
   */
  public static Tensor outer(Tensor tensorA, Tensor tensorB) {
    return new Tensor(BinaryOps.contract(tensorA.core, tensorB.core, new int[0], new int[0]));
  }

  
  /**
   * Adds another tensor to this tensor elementwise.
   *
   * @param tensorB the tensor to add.
   * @return a new tensor containing the elementwise sum.
   */
  public Tensor add(Tensor tensorB) {
    return add(this, tensorB);
  }

  /**
   * Adds two tensors elementwise.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor containing the elementwise sum.
   */
  public static Tensor add(Tensor tensorA, Tensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x + y, (x, y) -> 1.0, (x, y) -> 1.0);
  }

  /**
   * Subtracts another tensor from this tensor elementwise.
   *
   * @param tensorB the tensor to subtract.
   * @return a new tensor containing the elementwise difference.
   */
  public Tensor sub(Tensor tensorB) {
    return sub(this, tensorB);
  }

  /**
   * Subtracts one tensor from another elementwise.
   *
   * @param tensorA the base tensor.
   * @param tensorB the tensor to subtract.
   * @return a new tensor containing the elementwise difference.
   */
  public static Tensor sub(Tensor tensorA, Tensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x - y, (x, y) -> 1.0, (x, y) -> -1.0);
  }

  /**
   * Computes the Hadamard product (elementwise multiplication) between this
   * tensor and another.
   *
   * @param tensorB the tensor to multiply by.
   * @return a new tensor containing the elementwise product.
   */
  public Tensor hadamard(Tensor tensorB) {
    return hadamard(this, tensorB);
  }

  /**
   * Computes the Hadamard product (elementwise multiplication) between two
   * tensors.
   *
   * @param tensorA the first tensor.
   * @param tensorB the second tensor.
   * @return a new tensor containing the elementwise product.
   */
  public static Tensor hadamard(Tensor tensorA, Tensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x * y, (x, y) -> y, (x, y) -> x);
  }

  /**
   * Divides this tensor by another tensor elementwise.
   *
   * @param tensorB the denominator tensor.
   * @return a new tensor containing the elementwise quotient.
   */
  public Tensor div(Tensor tensorB) {
    return div(this, tensorB);
  }

  /**
   * Divides one tensor by another elementwise.
   *
   * @param tensorA the numerator tensor.
   * @param tensorB the denominator tensor.
   * @return a new tensor containing the elementwise quotient.
   */
  public static Tensor div(Tensor tensorA, Tensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x / y, (x, y) -> 1.0 / y, (x, y) -> -x / Math.pow(y, 2));
  }

  /**
   * Raises this tensor to the power of another tensor elementwise.
   *
   * @param tensorB the exponent tensor.
   * @return a new tensor containing the elementwise power results.
   */
  public Tensor pow(Tensor tensorB) {
    return pow(this, tensorB);
  }

  /**
   * Raises a base tensor to the power of an exponent tensor elementwise.
   *
   * @param tensorA the base tensor.
   * @param tensorB the exponent tensor.
   * @return a new tensor containing the elementwise power results.
   */
  public static Tensor pow(Tensor tensorA, Tensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> Math.pow(x, y), (x, y) -> y * Math.pow(x, y - 1), (x, y) -> Math.pow(x, y) * Math.log(x));
  }

  /**
   * Multiplies this tensor by another tensor using matrix multiplication.
   *
   * @param tensorB the multiplier tensor.
   * @return a new tensor containing the matrix product.
   */
  public Tensor matmul(Tensor tensorB) {
    return matmul(this, tensorB);
  }

  /**
   * Computes the matrix multiplication of two tensors.
   *
   * @param tensorA the first matrix tensor.
   * @param tensorB the second matrix tensor.
   * @return a new tensor containing the matrix product.
   */
  public static Tensor matmul(Tensor tensorA, Tensor tensorB) {
    Operation.verifyMatMul(tensorA.shape, tensorB.shape);

    int[] axesA = {tensorA.rank - 1};
    int[] axesB = {(tensorB.rank == 1) ? 0 : tensorB.rank - 2 };

    return contract(tensorA, tensorB, axesA, axesB);
  }

  // SHAPE OPERATIONS ---------------------------------

  /**
   * Broadcasts this tensor into a new target shape. Broadcasting a tensor is multiplying values of a tensor across
   * a new dimension to match the target shape layout. This is usually done implicitly and internally. This tensor must equal in rank
   * to the desired shape.
   *
   * @param shape the desired dimensions for the new tensor.
   * @return a new tensor matched to the target shape layout.
   */
  public Tensor broadcast(int... shape) {
    return broadcast(this, shape);
  }

  /**
   * Broadcasts a tensor into a new target shape. Broadcasting a tensor is multiplying values of a tensor across
   * a new dimension to match the target shape layout. This is usually done implicitly and internally.
   *
   * @param Tenor the tensor. Must equal in rank to the target shape.
   * @param shape the desired dimensions for the new tensor.
   * @return a new tensor matched to the target shape layout.
   */
  public static Tensor broadcast(Tensor tensor, int... shape) {
    return new Tensor(ShapeOps.broadcast(tensor.core, shape));
  }

  /**
   * Reshapes this tensor into a new target shape.
   *
   * @param shape the desired dimensions for the new tensor.
   * @return a new tensor matched to the target shape layout.
   */
  public Tensor reshape(int... shape) {
    return reshape(this, shape);
  }

  /**
   * Reshapes a specified tensor into a new target shape layout.
   *
   * @param tensor the tensor to reshape.
   * @param shape the desired dimensions for the new tensor.
   * @return a new tensor matched to the target shape layout.
   */
  public static Tensor reshape(Tensor tensor, int... shape) {
    return new Tensor(ShapeOps.reshape(tensor.core, shape));
  }

  /**
   * Permutes the axes order of this tensor according to the specified layout.
   *
   * @param axes the target ordering index map for the dimensions.
   * @return a new tensor with reordered axes.
   */
  public Tensor permute(int... axes) {
    return permute(this, axes);
  }

  /**
   * Permutes the axes order of a specified tensor according to the target layout.
   *
   * @param tensor the tensor to permute.
   * @param axes the target ordering index map for the dimensions.
   * @return a new tensor with reordered axes.
   */
  public static Tensor permute(Tensor tensor, int... axes) {
    Axes.verifyUniqueList("permutation", axes);
    CoreTensor out = ShapeOps.permute(tensor.core, axes);
    return new Tensor(out);
  }

  /**
   * Squeezes single-dimensional configurations out of this tensor's dimensions.
   *
   * @return a new tensor lacking single dimensions.
   */
  public Tensor squeeze() {
    return squeeze(this);
  }

  /**
   * Squeezes single-dimensional configurations out of a specified tensor's
   * dimensions.
   *
   * @param tensor the target tensor to squeeze.
   * @return a new tensor lacking single dimensions.
   */
  public static Tensor squeeze(Tensor tensor) {
    return new Tensor(ShapeOps.squeeze(tensor.core));
  }

  /**
   * Unsqueezes a single dimension into a specified tensor's structural layout. The tensor will be expanded by iterating over the shape, and if an axis of unsqueezing is detected, will pad the shape with a singleton dimension at that location.
   *
   * @param axes the location of the new dimensions to expand.
   * @return a new tensor expanded by an isolated dimension.
   */
  public Tensor unsqueeze(int... axes) {
    return unsqueeze(this, axes);
  }

  /**
   * Unsqueezes a single dimension into a specified tensor's structural layout. The tensor will be expanded by iterating over the shape, and if an axis of unsqueezing is detected, will pad the shape with a singleton dimension at that location.
   *
   * @param tensor the target tensor to expand.
   * @param axes the location of the new dimensions to expand.
   * @return a new tensor expanded by an isolated dimension.
   */
  public static Tensor unsqueeze(Tensor tensor, int... axes) {
    Axes.verifyUniqueList("unsqueeze", axes);

    return new Tensor(ShapeOps.unsqueeze(tensor.core, axes));
  }

  /**
   * Extracts a specific dimensional slice from this tensor layout.
   *
   * @param axis the dimension line targeted for slicing.
   * @param index the exact index position inside the targeted axis slice.
   * @return a new tensor representing the extracted slice.
   */
  public Tensor slice(int axis, int index) {
    return slice(this, axis, index);
  }

  /**
   * Extracts a specific dimensional slice from a specified tensor structure.
   *
   * @param tensor the tensor to slice from.
   * @param axis the dimension line targeted for slicing.
   * @param index the exact index position inside the targeted axis slice.
   * @return a new tensor representing the extracted slice.
   */
  public static Tensor slice(Tensor tensor, int axis, int index) {
    return new Tensor(ShapeOps.slice(tensor.core, axis, index));
  }

  /**
   * Stacks an array of source tensors uniformly together along a targeted axis.
   *
   * @param axis the axis line sequence configuration where tensors stack together.
   * @param tensors the target group of tensors being joined.
   * @return a new stacked tensor representation.
   */
  public static Tensor stack(int axis, Tensor... tensors) {
    CoreTensor[] coreTensors = new CoreTensor[tensors.length];

    for (int i = 0; i < tensors.length; i++) {
      coreTensors[i] = tensors[i].core;
    }

    return new Tensor(ShapeOps.stack(axis, coreTensors));
  }

  /**
   * Concatenates an array of source tensors together sequentially along a
   * targeted axis.
   *
   * @param axis the operational axis sequence where arrays join up.
   * @param tensors the target group of tensors being concatenated.
   * @return a new unified continuous structural tensor.
   */
  public static Tensor concat(int axis, Tensor... tensors) {
    CoreTensor[] coreTensors = new CoreTensor[tensors.length];

    for (int i = 0; i < tensors.length; i++) {
      coreTensors[i] = tensors[i].core;
    }
    return new Tensor(ShapeOps.concat(axis, coreTensors));
  }

  /**
   * Flattens this tensor into a single-dimensional layout.
   *
   * @return a new flattened tensor.
   */
  public Tensor flatten() {
    return flatten(this);
  }

  /**
   * Flattens a tensor into a single-dimensional layout.
   *
   * @param tensor the tensor to flatten.
   * @return a new flattened tensor.
   */
  public static Tensor flatten(Tensor tensor) {
    return reshape(tensor, -1);
  }

  /**
   * Transposes the dimensional axes of this tensor.
   *
   * @return a new transposed tensor.
   */
  public Tensor transpose() {
    return transpose(this);
  }

  /**
   * Transposes the dimensional axes of a tensor. In this implementation, 
   *
   * @param tensor the tensor to transpose.
   * @return a new transposed tensor.
   */
  public static Tensor transpose(Tensor tensor) {
    Geometry.verifyMinimumRank("transpose", 2, tensor.rank);

    int[] transposed = Tools.arrange(tensor.rank);

    // swapping the last 2 axes
    int temp = transposed[tensor.rank - 1];
    transposed[tensor.rank - 1] = transposed[tensor.rank - 2];
    transposed[tensor.rank - 2] = temp;

    Tensor out = permute(tensor, transposed);

    return out;
  }
  
  // REDUCTION OPERATIONS ---------------------------------

  /**
   * Sums the elements of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the sums.
   */
  public Tensor sum(int... axes) {
    return sum(this, axes);
  }

  /**
   * Sums the elements of a tensor along the specified axes.
   *
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the sums.
   */
  public static Tensor sum(Tensor tensor, int... axes) {
    return new Tensor(ReductionOps.sum(tensor.core, axes));
  }

  /**
   * Computes the product of the elements of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the products.
   */
  public Tensor prod(int... axes) {
    return prod(this, axes);
  }

  /**
   * Computes the product of the elements of a tensor along the specified axes.
   *
   * @param tensor the tensor to reduce.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the products.
   */
  public static Tensor prod(Tensor tensor, int... axes) {
    return new Tensor(ReductionOps.prod(tensor.core, axes));
  }

  /**
   * Finds the minimum values of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the minimums.
   */
  public Tensor min(int... axes) {
    return min(this, axes);
  }

  /**
   * Finds the minimum values of a tensor along the specified axes.
   *
   * @param tensor the tensor to evaluate.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the minimums.
   */
  public static Tensor min(Tensor tensor, int... axes) {
    return new Tensor(ReductionOps.min(tensor.core, axes));
  }

  /**
   * Finds the maximum values of this tensor along the specified axes.
   *
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the maximums.
   */
  public Tensor max(int... axes) {
    return max(this, axes);
  }

  /**
   * Finds the maximum values of a tensor along the specified axes.
   *
   * @param tensor the tensor to evaluate.
   * @param axes the dimensions to reduce.
   * @return a new reduced tensor containing the maximums.
   */
  public static Tensor max(Tensor tensor, int... axes) {
    return new Tensor(ReductionOps.max(tensor.core, axes));
  }

  // SHAPE OPERATIONS ---------------------------------


  // UNARY OPERATIONS ---------------------------------

  /**
   * Adds a scalar value to every element in this tensor.
   *
   * @param scalar the value to add.
   * @return a new shifted tensor.
   */
  public Tensor add(double scalar) {
    return add(this, scalar);
  }

  /**
   * Adds a scalar value to every element in a tensor.
   *
   * @param tensor the tensor to modify.
   * @param scalar the value to add.
   * @return a new shifted tensor.
   */
  public static Tensor add(Tensor tensor, double scalar) {
    return elementwise(tensor, (a) -> a + scalar, (a) -> 1.0);
  }

  /**
   * Multiplies every element in this tensor by a scalar value.
   *
   * @param scalar the scaling factor.
   * @return a new scaled tensor.
   */
  public Tensor mul(double scalar) {
    return mul(this, scalar);
  }

  /**
   * Multiplies every element in a tensor by a scalar value.
   *
   * @param tensor the tensor to scale.
   * @param scalar the scaling factor.
   * @return a new scaled tensor.
   */
  public static Tensor mul(Tensor tensor, double scalar) {
    return elementwise(tensor, (a) -> a * scalar, (a) -> scalar);
  }

  /**
   * Raises every element in this tensor to a scalar power.
   *
   * @param power the exponent value.
   * @return a new exponentiated tensor.
   */
  public Tensor pow(double power) {
    return pow(this, power);
  }

  /**
   * Raises every element in a tensor to a scalar power.
   *
   * @param tensor the base tensor.
   * @param scalar the exponent value.
   * @return a new exponentiated tensor.
   */
  public static Tensor pow(Tensor tensor, double power) {
    return elementwise(tensor, (a) -> Math.pow(a, power), (a) -> power * Math.pow(a, power - 1.0));
  }

  /**
   * Exponentiates every element in this tensor to a scalar power (scalar^element).
   *
   * @param base the base value.
   * @return a new exponentiated tensor.
   */
  public Tensor exp(double base) {
    return exp(this, base);
  }

  /**
   * Exponentiates every element in this tensor to a scalar power (scalar^element).
   *
   * @param tensor the exponent tensor.
   * @param base the base value.
   * @return a new exponentiated tensor.
   */
  public static Tensor exp(Tensor tensor, double base) {
    Argument.aboveValue(0, base, "exp");
    return elementwise(tensor, (a) -> Math.pow(base, a), (a) -> Math.pow(base, a) * Math.log(base));
  }

  /**
   * Exponentiates every element in this tensor.
   *
   * @return a new exponentiated tensor.
   */
  public Tensor exp() {
    return exp(this);
  }

  /**
   * Exponentiates every element in this tensor.
   *
   * @param tensor the base tensor.
   * @return a new exponentiated tensor.
   */
  public static Tensor exp(Tensor tensor) {
    return elementwise(tensor, (a) -> Math.exp(a), (a) -> Math.exp(a));
  }

  /**
   * Take the logarithm of every element in this tensor using a specified base.
   *
   * @return a new log tensor.
   */
  public Tensor log(double base) {
    return log(this, base);
  }

  /**
   * Exponentiates every element in this tensor using a specified base.
   *
   * @param tensor the base tensor.
   * @param base the base of the logarithm.
   * @return a new exponentiated tensor.
   */
  public static Tensor log(Tensor tensor, double base) {
    Operation.aboveValue(0, tensor.dump(), "log");
    Argument.aboveValue(0, base, "log");
    return elementwise(tensor, (a) -> Math.log(a)/Math.log(base), (a) -> 1.0 / (a * Math.log(base)));
  }

  /**
   * Take the logarithm of every element in this tensor using a specified base.
   *
   * @return a new log tensor.
   */
  public Tensor ln() {
    return ln(this);
  }

  /**
   * Exponentiates every element in this tensor using a specified base.
   *
   * @param tensor the base tensor.
   * @param base the base of the logarithm.
   * @return a new exponentiated tensor.
   */
  public static Tensor ln(Tensor tensor) {
    Operation.aboveValue(0, tensor.dump(), "ln (natural log)");
    return elementwise(tensor, (a) -> Math.log(a), (a) -> 1.0 / a);
  }

  /**
   * Returns the absolute value of each element in this tensor.
   *
   * @return a new tensor containing the absolute values.
   */
  public Tensor abs() {
    return abs(this);
  }

  /**
   * Returns the absolute value of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a new tensor containing the absolute values.
   */
  public static Tensor abs(Tensor tensor) {
    return elementwise(tensor, Math::abs, a -> (a > 0) ? 1.0 : -1.0);
  }
  /**
   * Returns the sign of each element in this tensor.
   *
   * @return a new tensor containing the signs.
   */
  public Tensor sign() {
    return sign(this);
  }

  /**
   * Returns the sign of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a new tensor containing the signs.
   */
  public static Tensor sign(Tensor tensor) {
    return elementwise(tensor, Math::signum, a -> 0.0);
  }

  /**
   * Returns the ceiling of each element in this tensor.
   *
   * @return a new tensor containing the ceilings.
   */
  public Tensor ceil() {
    return ceil(this);
  }

  /**
   * Returns the ceiling of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a new tensor containing the ceilings.
   */
  public static Tensor ceil(Tensor tensor) {
    return elementwise(tensor, Math::ceil, a -> 1.0);
  }

  /**
   * Returns the floor of each element in this tensor.
   *
   * @return a new tensor containing the floors.
   */
  public Tensor floor() {
    return floor(this);
  }

  /**
   * Returns the floor of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @return a new tensor containing the floors.
   */
  public static Tensor floor(Tensor tensor) {
    return elementwise(tensor, Math::floor, a -> 1.0);
  }

  /**
   * Returns the rounded value of each element in this tensor.
   *
   * @param scale the number of decimal places to round to.
   * @return a new tensor containing the rounded values.
   */
  public Tensor round(int scale) {
    return round(this, scale);
  }

  /**
   * Returns the rounded value of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @param scale the number of decimal places to round to.
   * @return a new tensor containing the rounded values.
   */
  public static Tensor round(Tensor tensor, int scale) {
    Argument.belowValue(0, scale, "rounding");
    return elementwise(tensor, a -> Miscellaneous.round(a, scale), a -> 1.0);
  }

  /**
   * Returns the truncated value of each element in this tensor.
   *
   * @param scale the number of decimal places to truncate to.
   * @return a new tensor containing the truncated values.
   */
  public Tensor trunc(int scale) {
    return trunc(this, scale);
  }

  /**
   * Returns the truncated value of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @param scale the number of decimal places to truncate to.
   * @return a new tensor containing the truncated values.
   */
  public static Tensor trunc(Tensor tensor, int scale) {
    Argument.belowValue(0, scale, "truncate");
    return elementwise(tensor, a -> Miscellaneous.truncate(a, scale), a -> 1.0);
  }

  /**
   * Returns the fractional part of each element in this tensor.
   *
   * @param scale the number of decimal places to keep for the fractional part.
   * @return a new tensor containing the fractional parts.
   */
  public Tensor frac(int scale) {
    return frac(this, scale);
  }

  /**
   * Returns the fractional part of each element in this tensor.
   *
   * @param tensor the tensor to evaluate.
   * @param scale the number of decimal places to keep for the fractional part.
   * @return a new tensor containing the fractional parts.
   */
  public static Tensor frac(Tensor tensor, int scale) {
    Argument.belowValue(0, scale, "fractional");
    return elementwise(tensor, a -> Miscellaneous.fractional(a, scale), a -> 1.0);
  }

  /**
   * Returns the clipped value of each element in this tensor.
   *
   * @param min the minimum value to clip to.
   * @param max the maximum value to clip to.
   * @return a new tensor containing the clipped values.
   */
  public Tensor clip(double min, double max) {
    return clip(this, min, max);
  }

  /**
   * Returns the clipped value of each element in this tensor.
   *  
   * @param tensor the tensor to evaluate.
   * @param min the minimum value to clip to.
   * @param max the maximum value to clip to.
   * @return a new tensor containing the clipped values.
   */
  public static Tensor clip(Tensor tensor, double min, double max) {
    Argument.aboveValue(min, max, "clipping");
    Argument.belowValue(max, min, "clipping");
    return elementwise(tensor, a -> Math.min(Math.max(a, min), max), a -> Unary.step(a-min) * Unary.step(-(a-max)));
  }
}

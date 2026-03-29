package com.aufy.jnet;

import com.aufy.jnet.tensor.core.backend.func.Binary;
import com.aufy.jnet.tensor.core.backend.func.Reduction;
import com.aufy.jnet.tensor.core.backend.func.Unary;
import com.aufy.jnet.tensor.core.backend.util.ArrayOps;
import com.aufy.jnet.tensor.core.exception.Calculus;
import com.aufy.jnet.tensor.core.exception.Geometry;
import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.graph.main.BinaryOps;
import com.aufy.jnet.tensor.graph.main.ReductionOps;
import com.aufy.jnet.tensor.graph.main.ShapeOps;
import com.aufy.jnet.tensor.graph.main.UnaryOps;

public class Tensor {
  private final CoreTensor core;

  public final int[] shape;
  public final int rank;
  public final int size;

  public boolean enableGrad;

  public Tensor(double[] data, int... shape){
    Geometry.verifyNotEmpty("initialization", shape);
    Geometry.verifyDataShape("initialization", data.length, shape);

    this.shape = shape;
    this.rank = shape.length;
    this.size = data.length;
    this.enableGrad = false;
    
    this.core = new CoreTensor(data, shape);
  }
  
  protected Tensor(CoreTensor tensor) {
    this.shape = tensor.shape.clone();
    this.rank = tensor.rank;
    this.size = tensor.size;
    this.enableGrad = false;

    this.core = tensor.clone();
  }

  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  @Override
  public String toString() {
    return core.toString();
  }

  // ########################################################################################################### //
  //                                                 AUTOGRAD                                                    //
  // ########################################################################################################### //
  
  public void backward() {
    Calculus.verifyDifferentiable("backwards", core.requiresGrad); 
    core.backward();
  }
  
  public void noGrad() {
  core.noGrad();}
  public static void noGrad(CoreTensor tensor) {tensor.noGrad();
  }

  public Tensor detach() {
  return new Tensor(CoreTensor.detach(this.core));}
  public static Tensor detach(Tensor tensor) {return new Tensor(CoreTensor.detach(tensor.core));
  }

  public void zeroGrad() {
  core.zeroGrad();}
  public static void zeroGrad(Tensor tensor) {tensor.zeroGrad();
  }
  
  // ########################################################################################################### //
  //                                          GENERAL OPERATIONS                                                 //
  // ########################################################################################################### //

  // BINARY OPERATIONS ---------------------------------
  
  public static Tensor elementwise(Tensor tensorA, Tensor tensorB, Binary operation, Binary dA, Binary dB) {
    Geometry.verifySizeMatch("elementwise operation", tensorA.shape, tensorB.shape);
    return new Tensor(BinaryOps.elementwise(tensorA.core, tensorB.core, operation, dA, dB));
  }
  
  public static Tensor contract(Tensor tensorA, Tensor tensorB, int[] axesA, int[] axesB) {
    Geometry.verifyAxis("tensor contraction", tensorA.rank, ArrayOps.max(axesA));
    Geometry.verifyAxis("tensor contraction", tensorB.rank, ArrayOps.max(axesB));

    return squeeze(new Tensor(BinaryOps.contract(tensorA.core, tensorB.core, axesA, axesB)));
  }

  // REDUCTION OPERATIONS ---------------------------------
  
  public static Tensor reduce(Tensor tensor, Reduction operation, int... axes) {
    return new Tensor(ReductionOps.reduce(tensor.core, operation, axes));
  }
  
  // SHAPE OPERATIONS ---------------------------------

  public static Tensor reshape(Tensor tensor, int... shape) {
    return new Tensor(ShapeOps.reshape(tensor.core, shape));
  }

  public static Tensor permute(Tensor tensor, int... axes) {
    return new Tensor(ShapeOps.permute(tensor.core, axes));
  }

  public static Tensor squeeze(Tensor tensor) {
    return new Tensor(ShapeOps.squeeze(tensor.core));
  }

  public static Tensor unsqueeze(Tensor tensor) {
    return new Tensor(ShapeOps.unsqueeze(tensor.core));
  }

  public static Tensor slice(Tensor tensor, int axis, int index) {
    return new Tensor(ShapeOps.slice(tensor.core, axis, index));
  }

  public static Tensor stack(int axis, Tensor... tensors) {
    CoreTensor[] coreTensors = new CoreTensor[tensors.length];

    for (int i = 0; i < tensors.length; i++) {
      coreTensors[i] = tensors[i].core;
    }
    
    return new Tensor(ShapeOps.stack(axis, coreTensors));
  }
  
  public static Tensor concat(int axis, Tensor... tensors) {
    CoreTensor[] coreTensors = new CoreTensor[tensors.length];
  
    for (int i = 0; i < tensors.length; i++) {
      coreTensors[i] = tensors[i].core;
    }
    return new Tensor(ShapeOps.concat(axis, coreTensors));
  }

  // UNARY OPERATIONS ---------------------------------

  public static Tensor apply(Tensor tensor, Unary operation, Unary derivative) {
    return new Tensor(UnaryOps.apply(tensor.core, operation, derivative));
  }
  
  // ########################################################################################################### //
  //                                         SPESIFIC OPERATIONS                                                 //
  // ########################################################################################################### //
  
  // BINARY OPERATIONS ---------------------------------

  public static Tensor add(Tensor tensorA, Tensor tensorB) {
    Geometry.verifySizeMatch("addition", tensorA.shape, tensorB.shape);
    return elementwise(tensorA, tensorB, (x, y) -> x + y, (x, y) -> 1.0, (x, y) -> 1.0);
  }

  public static Tensor sub(Tensor tensorA, Tensor tensorB) {
    Geometry.verifySizeMatch("subtraction", tensorA.shape, tensorB.shape);
    return elementwise(tensorA, tensorB, (x, y) -> x - y, (x, y) -> 1.0, (x, y) -> -1.0);
  }

  public static Tensor hadamard(Tensor tensorA, Tensor tensorB) {
    Geometry.verifySizeMatch("pointwise hadamard product", tensorA.shape, tensorB.shape);
    return elementwise(tensorA, tensorB, (x, y) -> x * y, (x, y) -> y, (x, y) -> x);
  }
  
  public static Tensor div(Tensor tensorA, Tensor tensorB) {
    Geometry.verifySizeMatch("pointwise division", tensorA.shape, tensorB.shape);
    return elementwise(tensorA, tensorB, (x, y) -> x / y, (x, y) -> 1.0 / y, (x, y) -> -x / Math.pow(y, 2));
  }
  
  public static Tensor pow(Tensor tensorA, Tensor tensorB) {
    Geometry.verifySizeMatch("pointwise power", tensorA.shape, tensorB.shape);
    return elementwise(tensorA, tensorB, (x, y) -> Math.pow(x, y), (x, y) -> y * Math.pow(x, y - 1), (x, y) -> Math.pow(x, y) * Math.log(x));
  }
  
  public static Tensor matmul(Tensor tensorA, Tensor tensorB) {
    Geometry.verifyMatMul(tensorA.shape, tensorB.shape);

    int[] axesA = {tensorA.rank - 1};
    int[] axesB = {(tensorB.rank == 1) ? 0 : tensorB.rank - 2};

    return contract(tensorA, tensorB, axesA, axesB);
  }

  // REDUCTION OPERATIONS ---------------------------------

  public static Tensor sum(Tensor tensor, int... axes) {
    return new Tensor(ReductionOps.sum(tensor.core, axes));
  }

  public static Tensor prod(Tensor tensor, int... axes) {
    return new Tensor(ReductionOps.prod(tensor.core, axes));
  }

  public static Tensor min(Tensor tensor, int... axes) {
    return new Tensor(ReductionOps.min(tensor.core, axes));
  }

  public static Tensor max(Tensor tensor, int... axes) {
    return new Tensor(ReductionOps.max(tensor.core, axes));
  }

  // SHAPE OPERATIONS ---------------------------------

  public static Tensor flatten(Tensor tensor) {
    return reshape(tensor, -1);
  }

  public static Tensor transpose(Tensor tensor) {
    return permute(tensor, -1);
  }

  // UNARY OPERATIONS ---------------------------------

  public static Tensor add(Tensor tensor, double scalar) {
    return apply(tensor, (a) -> a + scalar, (a) -> 1.0);
  }
  
  public static Tensor mul(Tensor tensor, double scalar) {
    return apply(tensor, (a) -> a * scalar, (a) -> scalar);
  }
  
  public static Tensor pow(Tensor tensor, double scalar) {
    return apply(tensor, (a) -> Math.pow(a, scalar), (a) -> scalar * Math.pow(a, scalar - 1.0));
  }
  
}

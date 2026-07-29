package com.aufy.jnet.core.tensor.core.implementations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

import com.aufy.jnet.core.backend.exceptions.internal.Data;
import com.aufy.jnet.core.tensor.core.backend.util.ArrayTools;
import com.aufy.jnet.core.tensor.graph.init.TensorCoreGenerator;

public class CoreTensor {
  public static boolean verbose = false;
  public RawTensor core;
  
  public int rank;
  public int size;
  public int[] shape;
  
  public boolean requiresGrad;
  public CoreTensor grad; // stores d[end]/d[this]
  public List<CoreTensor> parents = new ArrayList<>();
  public Consumer<CoreTensor> derivative; // coretensor -> coretensor that gives the [grad] values to its parents
  
  public CoreTensor(double[] data, int... shape) {
    this(new RawTensor(data, shape));
  }

  public CoreTensor(RawTensor core) {
    this.core = core;
    this.requiresGrad = false;
    this.shape = core.getShape();
    this.size = core.getSize();
    this.rank = core.getRank();
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
  public static CoreTensor ones(int... shape) {
    return TensorCoreGenerator.ones(shape);
  }

  /**
   * Creates a tensor filled with ones according to the shape of this tensor. This tensor will not be modified.
   * 
   * @return the tensor filled with ones.
   */
  public CoreTensor onesLike() {
    return ones(this.shape);
  }

  /**
   * Creates a tensor filled with ones according to the shape of another tensor.
   * 
   * @param tensor the tensor.
   * @return the tensor filled with ones.
   */
  public static CoreTensor onesLike(CoreTensor tensor) {
    return ones(tensor.shape);
  }

  /**
   * Creates a tensor filled with ones.
   * 
   * @param shape the shape of the tensor.
   * @return the tensor filled with ones.
   */
  public static CoreTensor zeros(int... shape) {
    return TensorCoreGenerator.zeros(shape);
  }

  /**
   * Creates a tensor filled with zeros according to the shape of this tensor. This tensor will not be modified.
   * 
   * @return the tensor filled with zeros.
   */
  public CoreTensor zerosLike() {
    return zeros(this.shape);
  }

  /**
   * Creates a tensor filled with zeros according to the shape of another tensor.
   * 
   * @param tensor the tensor.
   * @return the tensor filled with zeros.
   */
  public static CoreTensor zerosLike(CoreTensor tensor) {
    return zeros(tensor.shape);
  }
  
  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  public CoreTensor detach() {
    return detach(this);
  }

  public static CoreTensor detach(CoreTensor tensor) {
    CoreTensor out = new CoreTensor(tensor.core);
    out.requiresGrad = false;
    return out;
  }
  
  public CoreTensor duplicate() {return duplicate(this);}
  public static CoreTensor duplicate(CoreTensor tensor) {
    CoreTensor out = new CoreTensor(tensor.core.duplicate());

    // copy CoreTensor attributes
    out.requiresGrad = tensor.requiresGrad;
    out.grad = tensor.grad;
    out.parents = tensor.parents;
    out.derivative = tensor.derivative;

    return out;
  }

  public double[] dump() {
    return this.core.dump();
  }

  public CoreTensor noGrad() {return noGrad(this);}
  public static CoreTensor noGrad(CoreTensor tensor) {
    tensor.requiresGrad = false;
    tensor.parents = null;
    tensor.derivative = null;

    return tensor;
  }

  public double get(int... indices) {
    return this.core.get(indices);
  }

  @Override
  public String toString() {
    if (this.core.dump() == null || this.shape == null || this.core.dump().length == 0) return this.getClass().getSimpleName() + "[null]";

    String prefix = this.getClass().getSimpleName() + Arrays.toString(this.shape) + "(\n";
    String content = ArrayTools.print(this.core.dump(), this.shape, this.core.getStrides(), 0, 0, 2);

    if (verbose) {
      return prefix + content + "\n\n" + " grad=" + this.requiresGrad + "\n" + " parents=" + this.parents.size() + "\n" + " derivative=" + this.derivative + "\n)";
    } else {
      return prefix + content + "\n)";
    }
  }

  private void addInPlace(RawTensor other) {
    Data.verifyEqualSizes("Data transfer", this.dump(), other.dump());
    
    // Perform raw array addition without creating a new result array
    for (int i = 0; i < this.dump().length; i++) {
      this.core.data[i] += other.data[i];
    }
  }

  // ########################################################################################################### //
  //                                              AUTOGRAD LOGIC                                                 //
  // ########################################################################################################### //

  public void zeroGrad() {CoreTensor.zeroGrad(this);}
  public static void zeroGrad(CoreTensor tensor) {
    List<CoreTensor> nodes = buildGraph(tensor);
    for (CoreTensor node : nodes) {
      if (node.grad != null) {
        java.util.Arrays.fill(node.grad.core.dump(), 0.0);
      }
    }
  }

  public void accumulate(CoreTensor incomingGrad) {
    if (!this.requiresGrad) return;
    
    if (this.grad == null) {
      // initialize with zeros of the same shape as the data
      this.grad = TensorCoreGenerator.zerosLike(this);
    }

    this.grad.addInPlace(incomingGrad.core);
  }

  public static List<CoreTensor> buildGraph(CoreTensor root) {
    List<CoreTensor> order = new ArrayList<>();
    Set<CoreTensor> visited = new HashSet<>();
    
    visit(root, visited, order);
    
    // reverse the order because we want to go from out --> in
    Collections.reverse(order);
    return order;
  }

  private static void visit(CoreTensor node, Set<CoreTensor> visited, List<CoreTensor> order) {
    if (node == null || visited.contains(node)) return;
    
    visited.add(node);
    if (node.parents != null) {
      for (CoreTensor parent : node.parents) {
        visit(parent, visited, order);
      }
    }
    order.add(node);
  }

  public void backward() {
    this.grad = TensorCoreGenerator.onesLike(this);

    List<CoreTensor> order = buildGraph(this);

    for (CoreTensor node : order) {
      if (node.derivative != null && node.grad != null) {
        node.derivative.accept(node.grad);
      }
    }
  }

}
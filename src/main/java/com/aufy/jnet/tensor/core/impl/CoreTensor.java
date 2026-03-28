package com.aufy.jnet.tensor.core.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import com.aufy.jnet.tensor.core.backend.util.ArrayTools;
import com.aufy.jnet.tensor.graph.init.TensorCoreGenerator;

public class CoreTensor{
  /*
  unlike rawtensor, Tensor would inherit this so these public attributes are huge problems; need a way to hide these
  from the user while keeping the engine clean and usable
  */

  public static boolean verbose = false;
  public int rank;
  public int size;
  public int[] shape;
  public boolean requiresGrad;
  public CoreTensor grad;
  
  public RawTensor core;
  public List<CoreTensor> parents = new ArrayList<>();
  public Consumer<CoreTensor> derivative;
  public int[] allAxes;
  
  public CoreTensor(double[] data, int... shape) {
    this(new RawTensor(data, shape));
  }

  public CoreTensor(RawTensor core) {
    this.core = core;
    this.requiresGrad = false;
    this.shape = core.getShape();
    this.size = core.getSize();
    this.rank = core.getRank();
    this.allAxes = IntStream.range(0, this.rank + 1).toArray();
  }

  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  public static CoreTensor detach(CoreTensor tensor) {
    CoreTensor out = new CoreTensor(tensor.core);
    out.requiresGrad = false;
    return out;
  }
  
  public static CoreTensor clone(CoreTensor tensor) {
    return new CoreTensor(tensor.dump(), tensor.shape);
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
    if (this.core.dump() == null || this.shape == null || this.core.dump().length == 0) return "TensorCore[null]";

    String prefix = "TensorCore" + Arrays.toString(this.shape) + "(\n";
    String content = ArrayTools.print(this.core.dump(), this.shape, this.core.getStrides(), 0, 0, 2);
    String suffix = " grad=" + this.requiresGrad;

    if (verbose) {
      return prefix + content + "\n\n" + suffix + "\n)";
    } else {
      return prefix + content + "\n)";
    }
  }

  private void addInPlace(RawTensor other) {
    if (this.dump().length != other.dump().length) {
      throw new IllegalArgumentException("Internal data size mismatch for in-place add.");
    }
    
    // Perform raw array addition without creating a new result array
    for (int i = 0; i < this.dump().length; i++) {
      this.core.rawData()[i] += other.rawData()[i];
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
        java.util.Arrays.fill(node.grad.core.rawData(), 0.0);
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
    if (this.grad == null) {
      this.grad = TensorCoreGenerator.onesLike(this);
    } else {
      java.util.Arrays.fill(this.grad.core.rawData(), 1.0);
    }

    List<CoreTensor> order = buildGraph(this);

    for (CoreTensor node : order) {
      if (node.derivative != null && node.grad != null) {
        node.derivative.accept(node.grad);
      }
    }
  }
}
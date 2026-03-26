package com.aufy.jnet.tensor.core.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import com.aufy.jnet.tensor.core.backend.util.ArrayOps;
import com.aufy.jnet.tensor.graph.init.TensorCoreGenerator;

public class TensorCore implements Traceable{
  public static boolean verbose = false;
  public final int rank;
  public final int size;
  public final int[] shape; // prevents array switching but does not prevent direct modifications
  public boolean requiresGrad;
  public TensorCore grad;
  
  public final DataContainer core;
  public List<TensorCore> parents = new ArrayList<>();
  public Consumer<TensorCore> derivative;
  public int[] allAxes;
  
  public TensorCore(double[] data, int... shape) {
    this(new DataContainer(data, shape));
  }

  public TensorCore(DataContainer core) {
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

  @Override public List<TensorCore> getParents() { return Collections.unmodifiableList(this.parents); }
  @Override public int[] getShape() { return this.shape.clone(); }
  @Override public Consumer<TensorCore> getGradFunc() { return this.derivative; }

  public TensorCore detach() {return detach(this);}
  public static TensorCore detach(TensorCore tensor) {
    TensorCore out = new TensorCore(tensor.core);
    out.requiresGrad = false;
    return out;
  }
  
  public TensorCore clone() {return TensorCore.clone(this);}
  public static TensorCore clone(TensorCore tensor) {
    return new TensorCore(tensor.dump(), tensor.shape);
  }

  public double[] dump() {
    return this.core.dump();
  }

  public TensorCore noGrad() {return noGrad(this);}
  public static TensorCore noGrad(TensorCore tensor) {
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
    String content = ArrayOps.print(this.core.dump(), this.shape, this.core.getStrides(), 0, 0, 2);
    String suffix = " grad=" + this.requiresGrad;

    if (verbose) {
      return prefix + content + "\n\n" + suffix + "\n)";
    } else {
      return prefix + content + "\n)";
    }
  }

  private void addInPlace(DataContainer other) {
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

  public void zeroGrad() {TensorCore.zeroGrad(this);}
  public static void zeroGrad(TensorCore tensor) {
    List<TensorCore> nodes = buildGraph(tensor);
    for (TensorCore node : nodes) {
      if (node.grad != null) {
        java.util.Arrays.fill(node.grad.core.rawData(), 0.0);
      }
    }
  }

  public void accumulate(TensorCore incomingGrad) {
    if (!this.requiresGrad) return;
    
    if (this.grad == null) {
      // initialize with zeros of the same shape as the data
      this.grad = TensorCoreGenerator.zerosLike(this);
    }

    this.grad.addInPlace(incomingGrad.core);
  }

  public static List<TensorCore> buildGraph(TensorCore root) {
    List<TensorCore> order = new ArrayList<>();
    Set<TensorCore> visited = new HashSet<>();
    
    visit(root, visited, order);
    
    // reverse the order because we want to go from out --> in
    Collections.reverse(order);
    return order;
  }

  private static void visit(TensorCore node, Set<TensorCore> visited, List<TensorCore> order) {
    if (node == null || visited.contains(node)) return;
    
    visited.add(node);
    if (node.parents != null) {
      for (TensorCore parent : node.parents) {
        visit(parent, visited, order);
      }
    }
    order.add(node);
  }

  public void backward() {
    if (!this.requiresGrad) {
      throw new IllegalStateException(".backward() cannot be called on a tensor that does not require gradients");
    } 

    if (this.parents == null && this.derivative == null) {
      throw new IllegalStateException("This tensor is the result of a non-differentiable operation");
    }

    if (this.grad == null) {
      this.grad = TensorCoreGenerator.onesLike(this);
    } else {
      java.util.Arrays.fill(this.grad.core.rawData(), 1.0);
    }

    List<TensorCore> order = buildGraph(this);

    for (TensorCore node : order) {
      if (node.derivative != null && node.grad != null) {
        node.derivative.accept(node.grad);
      }
    }
  }
}
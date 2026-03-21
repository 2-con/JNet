import java.util.Arrays;
import java.util.List;

import tensor.core.Engine;
import tensor.core.Generator;
import tensor.operation.Binary;
import tensor.operation.Unary;
import tensor.operation.Reduction;
import tensor.RawTensor;

public class Tensor {
  public final RawTensor container;
  public boolean requiresGrad;
  public Tensor grad;
  
  private List<Tensor> parents;
  private BackwardOp gradFn; // A simple interface or lambda

  
  public Tensor(RawTensor container) {
    this.container = container;
    this.requiresGrad = false;
  }

  public Tensor(double[] data, int... shape) {
    this(new RawTensor(data, shape));
  }


  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // TENSOR GENERATORS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  public Tensor add(Tensor other) {
    // 1. Backend call
    RawTensor resData = this.container.add(other.container);
    Tensor res = new Tensor(resData);

    // 2. Autograd Setup
    if (this.requiresGrad || other.requiresGrad) {
      res.requiresGrad = true;
      res.parents = List.of(this, other);
      res.gradFn = (g) -> {
        if (this.requiresGrad) this.accumulate(g);
        if (other.requiresGrad) other.accumulate(g);
      };
    }
    return res;
  }

  public Tensor sin() {
    Tensor res = new Tensor(this.container.apply(Math::sin));
    if (this.requiresGrad) {
      res.requiresGrad = true;
      res.parents = List.of(this);
      res.gradFn = (g) -> this.accumulate(g.mul(this.container.apply(Math::cos)));
    }
    return res;
  }


  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // TENSOR GENERATORS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // TENSOR GENERATORS
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  public void backward() {
    if (this.grad == null) {
      this.grad = Tensor.onesLike(this);
    }
    
    // 1. Topological Sort
    List<Tensor> order = Engine.topologicalSort(this);
    
    // 2. Propagate
    for (Tensor node : order) {
      if (node.gradFn != null) {
        node.gradFn.apply(node.grad);
      }
    }
  }

  private void accumulate(Tensor g) {
    if (this.grad == null) this.grad = Tensor.zerosLike(this);
    this.grad = this.grad.add(g); // Summing gradients for the chain rule
  }
}
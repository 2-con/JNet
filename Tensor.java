import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import tensor.core.Engine;
import tensor.core.Generator;
import tensor.ops.Backwards;
import tensor.ops.Binary;
import tensor.ops.Reduction;
import tensor.ops.Unary;
import tensor.TensorCore;
import tensor.tools.ArrayTools;
import tensor.tools.Statistics;

public class Tensor {
  public boolean requiresGrad;
  public Tensor grad; 
  public final int rank;
  public final int size;
  
  private List<Tensor> parents = new ArrayList<>();
  private Backwards gradFunc;
  
  private final TensorCore core;
  private final int[] shape;
  
  public Tensor(double[] data, int... shape) {
    this(new TensorCore(data, shape));
  }

  public Tensor(TensorCore core) {
    this.core = core;
    this.requiresGrad = true;
    this.shape = core.getShape();
    this.size = core.getSize();
    this.rank = core.getRank();
  }
  
  // ########################################################################################################### //
  //                                              TENSOR GENERATORS                                              //
  // ########################################################################################################### //

  /**
   * Returns a Tensor with random values uniformly distributed between 0 and 1.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and uniformly distributed values
   */
  public static Tensor randomUniform(int... shape) {return new Tensor(Generator.generateUniform(shape), shape);}

  /**
   * Returns a Tensor with random values drawn from a standard normal distribution.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and standard normally distributed values
   */
  public static Tensor randomNormal(int... shape) {return new Tensor(Generator.generateGaussian(shape), shape);}

  /**
   * Returns a Tensor with random values drawn from an exponential distribution.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and exponentially distributed values
   */
  public static Tensor randomExponential(int... shape) {return new Tensor(Generator.generateExponential(shape), shape);}

  /**
   * Returns a Tensor with all elements set to zero.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and all elements set to zero
   */
  public static Tensor zeros(int... shape) {return new Tensor(Generator.zeros(shape), shape);}

  /**
   * Returns a Tensor with all elements set to zero, with the same shapes as the given tensor.
   * 
   * @param tensor the tensor to return a zeros Tensor for
   * @return a new Tensor with the same shapes as the given tensor and all elements set to zero
   */
  public static Tensor zerosLike(Tensor tensor) {return zeros(tensor.shape);}

  /**
   * Returns a Tensor with all elements set to one.
   * 
   * @param shape the shapes of the Tensor
   * @return a new Tensor with the given shapes and all elements set to one
   */
  public static Tensor ones(int... shape) {return new Tensor(Generator.ones(shape), shape);}

  /**
   * Returns a Tensor with all elements set to one, with the same shapes as the given tensor.
   * 
   * @param tensor the tensor to return a ones Tensor for
   * @return a new Tensor with the same shapes as the given tensor and all elements set to one
   */
  public static Tensor onesLike(Tensor tensor) {return ones(tensor.shape);}
  
  // ########################################################################################################### //
  //                                            CORE OPERATION METHODS                                           //
  // ########################################################################################################### //

  // SPECIALS

  public static Tensor permute(Tensor tensor, int... axes) {
    TensorCore resultCore = TensorCore.permute(tensor.core, axes);
    Tensor out = new Tensor(resultCore);

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      Tensor input = tensor;
      int[] inverse = new int[axes.length];

      for (int i = 0; i < axes.length; i++) {
        inverse[axes[i]] = i;
      }

      out.parents = List.of(input);

      out.gradFunc = (gradOutput) -> {

          TensorCore gradInputCore =
              TensorCore.permute(gradOutput.core, inverse);

          input.accumulate(new Tensor(gradInputCore));
      };
    }

    return out;
  }

  // UNARY

  /**
   * Applies a reduction operation elements specified in the axes of the given tensor. By default, the output tensor cannot automatically compute gradients
   * for this arbitrary reduction operation.
   * 
   * @param tensor the input tensor
   * @param operation the reduction operation to apply
   * @param axes the axes to apply the reduction operation by
   * @return a new Tensor containing the results with gradients set to false
   */
  public static Tensor reduce(Tensor tensor, Reduction operation, int... axes) {
    TensorCore resultCore = TensorCore.reduce(tensor.core, operation, axes);
    Tensor out = new Tensor(resultCore);

    out.requiresGrad = false;

    return out;
  }

  /**
   * Applies an elementwise unary operation to all elements of a tensor.
   * 
   * @param tensor the input tensor
   * @param operation the unary operation to apply
   * @param derivative the derivative of the operation with respect to the input
   * @return a new Tensor containing the result of applying the operation to each element of the input tensor
   */
  public static Tensor apply(Tensor tensor, Unary operation, Unary derivative) {
    TensorCore resultCore = TensorCore.apply(tensor.core, operation);
    Tensor out = new Tensor(resultCore);

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      Tensor input = tensor;
      out.parents = List.of(input);

      out.gradFunc = (gradOutput) -> {
        TensorCore derivativeCore = TensorCore.apply(input.core, derivative);
        TensorCore gradInputCore = TensorCore.combine(gradOutput.core, derivativeCore, (a, b) -> a * b);
        input.accumulate(new Tensor(gradInputCore));
      };
    }

    return out;
  }

  // BINARY

  /**
   * Applies a binary operation to all elements of two tensors.
   * 
   * @param a the first tensor
   * @param b the second tensor
   * @param op the binary operation to apply
   * @param dA the derivative of the operation with respect to the first input (Tensor a)
   * @param dB the derivative of the operation with respect to the second input (Tensor b)
   * @return a new Tensor containing the result of applying the operation to each element of the input tensors
   */
  public static Tensor combine(Tensor a, Tensor b, Binary op, Binary dA, Binary dB) {
    TensorCore resultCore = TensorCore.combine(a.core, b.core, op);
    Tensor out = new Tensor(resultCore);

    out.requiresGrad = a.requiresGrad || b.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(a, b);

      out.gradFunc = (gradOutput) -> {
        if (a.requiresGrad) {
          TensorCore dACore = TensorCore.combine(a.core, b.core, dA);
          TensorCore gradA  = TensorCore.combine(gradOutput.core, dACore, (x, y) -> x * y);
          a.accumulate(new Tensor(gradA));
        }

        if (b.requiresGrad) {
          TensorCore dBCore = TensorCore.combine(a.core, b.core, dB);
          TensorCore gradB  = TensorCore.combine(gradOutput.core, dBCore, (x, y) -> x * y);
          b.accumulate(new Tensor(gradB));
        }
      };
    }

    return out;
  }

  /**
   * Contracts two tensors along the specified axes.
   * 
   * @param a the first tensor
   * @param b the second tensor
   * @param axesA the axes of the first tensor to contract along
   * @param axesB the axes of the second tensor to contract along
   * @return a new Tensor containing the result of contracting the two input tensors along the specified axes
   */
  public static Tensor contract(Tensor a, Tensor b, int[] axesA, int[] axesB) {
    TensorCore resultCore = TensorCore.contract(a.core, b.core, axesA, axesB);

    Tensor out = new Tensor(resultCore);
    out.requiresGrad = a.requiresGrad || b.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(a, b);

      // returns the grad axes of A and B
      int[][] survivors = Engine.getResultAxes(a.shape, b.shape, axesA, axesB);

      out.gradFunc = (gradOutput) -> {
        if (a.requiresGrad) {
          TensorCore gradACore = TensorCore.contract(gradOutput.core, b.core, survivors[1], axesB);
          a.accumulate(new Tensor(gradACore));
        }

        if (b.requiresGrad) {
          TensorCore gradBCore = TensorCore.contract(a.core, gradOutput.core, axesA, survivors[0]);
          b.accumulate(new Tensor(gradBCore));
        }
      };
    }

    return out;
  }
  
  // ########################################################################################################### //
  //                                                EXTRA METHODS                                                //
  // ########################################################################################################### //
  
  // REDUCTION

  public Tensor sum(int... axes) {return sum(this, axes);}
  public static Tensor sum(Tensor tensor, int... axes) {
    Tensor out = new Tensor(TensorCore.reduce(tensor.core, ArrayTools::sum, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      Tensor input = tensor;

      out.parents = List.of(input);

      out.gradFunc = (gradOutput) -> {
        double[] gradInput = Engine.broadcastGrad(gradOutput.flatten(), gradOutput.shape, input.shape, axes);
        input.accumulate(new Tensor(gradInput, input.shape));
      };
    }

    return out;
  }

  public Tensor prod(int... axes) {return prod(this, axes);}
  public static Tensor prod(Tensor tensor, int... axes) {
    Tensor out = new Tensor(TensorCore.reduce(tensor.core, ArrayTools::prod, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.gradFunc = (gradOutput) -> {
        double[] gradExpanded = Engine.broadcastGrad(gradOutput.flatten(), gradOutput.shape, tensor.shape, axes);
        double[] yExpanded = Engine.broadcastGrad(out.flatten(), out.shape, tensor.shape, axes);

        double[] xData = tensor.flatten();
        double[] gradInput = new double[tensor.size];

        for (int i = 0; i < gradInput.length; i++) {
          gradInput[i] = gradExpanded[i] * (yExpanded[i] / xData[i]);
        }

        tensor.accumulate(new Tensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  public Tensor max(int... axes) {return max(this, axes);}
  public static Tensor max(Tensor tensor, int... axes) {
    Tensor out = new Tensor(TensorCore.reduce(tensor.core, Statistics::max, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.gradFunc = (gradOutput) -> {
        double[] gradExpanded = Engine.broadcastGrad(gradOutput.flatten(), gradOutput.shape, tensor.shape, axes);
        double[] yExpanded = Engine.broadcastGrad(out.flatten(), out.shape, tensor.shape, axes);

        double[] xData = tensor.flatten();
        double[] gradInput = new double[tensor.size];

        // get the index of the max to submit all the gradients to
        for (int i = 0; i < gradInput.length; i++) {
          if (xData[i] == yExpanded[i]) {
            gradInput[i] = gradExpanded[i];
          }
        }

        tensor.accumulate(new Tensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  public Tensor min(int... axes) {return min(this, axes);}
  public static Tensor min(Tensor tensor, int... axes) {
    Tensor out = new Tensor(TensorCore.reduce(tensor.core, Statistics::min, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.gradFunc = (gradOutput) -> {
        double[] gradExpanded = Engine.broadcastGrad(gradOutput.flatten(), gradOutput.shape, tensor.shape, axes);
        double[] yExpanded = Engine.broadcastGrad(out.flatten(), out.shape, tensor.shape, axes);

        double[] xData = tensor.flatten();
        double[] gradInput = new double[tensor.size];

        // get the index of the max to submit all the gradients to
        for (int i = 0; i < gradInput.length; i++) {
          if (xData[i] == yExpanded[i]) {
            gradInput[i] = gradExpanded[i];
          }
        }

        tensor.accumulate(new Tensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  public Tensor mean(int... axes) {return mean(this, axes);}
  public static Tensor mean(Tensor tensor, int... axes) {
    Tensor sum = sum(tensor, axes);
    int reductionSize = Engine.sizeOf(Engine.getSubShape(tensor.shape, axes));

    return mul(sum, 1.0 / reductionSize);
  }

  public Tensor variance(int... axes) {return variance(this, axes);}
  public static Tensor variance(Tensor tensor, int... axes) {
    Tensor mean = mean(tensor, axes);

    Tensor centered = Tensor.sub(tensor, mean);
    Tensor sq = Tensor.pow(centered, 2);

    return mean(sq, axes);
  }

  public Tensor stdev(int... axes) {return stdev(this, axes);}
  public static Tensor stdev(Tensor tensor, int... axes) {
    return Tensor.pow(Tensor.variance(tensor, axes), 0.5);
  }

  public Tensor skew(int... axes) {return skew(this, axes);}
  public static Tensor skew(Tensor tensor, int... axes) {
    Tensor mean = mean(tensor, axes);
    Tensor centered = sub(tensor, mean);

    Tensor m3 = mean(pow(centered, 3), axes);
    Tensor stdev = stdev(tensor, axes);

    return div(m3, pow(stdev, 3));
  }

  public Tensor kurtosis(int... axes) {return kurtosis(this, axes);}
  public static Tensor kurtosis(Tensor tensor, int... axes) {
    Tensor mean = mean(tensor, axes);
    Tensor centered = sub(tensor, mean);

    Tensor m4 = mean(pow(centered, 4), axes);
    Tensor stdev = stdev(tensor, axes);

    return div(m4, pow(stdev, 4));
  }

  // non-differentiables

  public Tensor range(int... axes) {return range(this, axes);}
  public static Tensor range(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("range() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::range, axes)).noGrad();
  }

  public Tensor median(int... axes) {return range(this, axes);}
  public static Tensor median(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("median() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::median, axes)).noGrad();
  }

  public Tensor mode(int... axes) {return range(this, axes);}
  public static Tensor mode(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("mode() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::mode, axes)).noGrad();
  }

  public Tensor quartile1(int... axes) {return range(this, axes);}
  public static Tensor quartile1(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("quartile1() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::quartile1, axes)).noGrad();
  }

  public Tensor quartile3(int... axes) {return range(this, axes);}
  public static Tensor quartile3(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("quartile3() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::quartile3, axes)).noGrad();
  }

  public Tensor iqr(int... axes) {return range(this, axes);}
  public static Tensor iqr(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("iqr() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::iqr, axes)).noGrad();
  }

  // UNARY

  public static Tensor add(Tensor tensor, double scalar) {
    return Tensor.apply(tensor, (a) -> a + scalar, (a) -> 1.0);
  }

  public static Tensor mul(Tensor tensor, double scalar) {
    return Tensor.apply(tensor, (a) -> a * scalar, (a) -> scalar);
  }
  
  public static Tensor pow(Tensor tensor, double scalar) {
    return Tensor.apply(tensor, (a) -> Math.pow(a, scalar), (a) -> scalar * Math.pow(a, scalar - 1.0));
  }

  // BINARY

  public static Tensor add(Tensor a, Tensor b) {
    return Tensor.combine(a, b, (x, y) -> x + y, (x, y) -> 1.0, (x, y) -> 1.0);
  }

  public static Tensor sub(Tensor a, Tensor b) {
    return Tensor.combine(a, b, (x, y) -> x - y, (x, y) -> 1.0, (x, y) -> 1.0);
  }

  public static Tensor hadamard(Tensor a, Tensor b) {
    return Tensor.combine(a, b, (x, y) -> x * y, (x, y) -> y, (x, y) -> x);
  }

  public static Tensor div(Tensor a, Tensor b) {
    return Tensor.combine(a, b, (x, y) -> x / y, (x, y) -> 1.0 / y, (x, y) -> 1.0 / x);
  }

  public static Tensor pow(Tensor a, Tensor b) {
    return Tensor.combine(a, b, (x, y) -> Math.pow(x, y), (x, y) -> 1.0 / (y * Math.pow(x, y - 1)), (x, y) -> Math.pow(x, y) * Math.log(x));
  }

  // function

  public static Tensor rescaleStandard(Tensor tensor){
    double mean = Statistics.mean(tensor.core.flatten());
    double stdDev = Statistics.stdev(tensor.core.flatten());

    return apply(tensor, n -> (n - mean) / stdDev, n -> 1.0 / stdDev);
  }

  public static Tensor rescaleMinMax(Tensor tensor){
    double max = Statistics.max(tensor.core.flatten());
    double min = Statistics.min(tensor.core.flatten());

    return apply(tensor, n -> (n - min) / (max - min), n -> 1.0 / (max - min));
  }
  
  public static Tensor rescaleRobust(Tensor tensor){
    double median = Statistics.median(tensor.core.flatten());
    double iqr = Statistics.iqr(tensor.core.flatten());
    
    return apply(tensor, n -> (n - median) / iqr, n -> 1.0 / iqr);
  }
  
  public static Tensor rescaleMaxAbs(Tensor tensor){
    double maxAbs = Statistics.max(ArrayTools.foreach(tensor.core.flatten(), n -> Math.abs(n)));
    
    return apply(tensor, n -> n / maxAbs, n -> 1.0 / maxAbs);
  }

  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  /**
   * Disables gradient computation for this tensor. Cannot be undone and changes the instance itself
   * 
   * @return the same tensor
   */
  public Tensor noGrad() {return noGrad(this);}

  /**
   * Disables gradient computation for a given tensor. Cannot be undone and changes the instance itself
   * 
   * @param tensor the tensor to disable gradient computation for
   * @return the same tensor
   */
  public static Tensor noGrad(Tensor tensor) {
    tensor.requiresGrad = false;
    tensor.parents = null;
    tensor.gradFunc = null;

    return tensor;
  }

  public double get(int... indices) {
    return this.core.get(indices);
  }

  @Override
  public String toString() {
    return "Tensor" + this.core.prettyPrint();
  }

  // ########################################################################################################### //
  //                                              AUTOGRAD LOGIC                                                 //
  // ########################################################################################################### //

  public void accumulate(Tensor incomingGrad) {
    if (!this.requiresGrad) return;
    
    if (this.grad == null) {
      // Initialize with zeros of the same shape as the data
      this.grad = Tensor.zerosLike(this);
    }
    // Use your Backend's combine method to add the gradients
    this.grad = new Tensor(TensorCore.combine(this.grad.core, incomingGrad.core, (a, b) -> a + b));
  }

  public static List<Tensor> buildGraph(Tensor root) {
    List<Tensor> order = new ArrayList<>();
    Set<Tensor> visited = new HashSet<>();
    
    // Internal recursive helper
    visit(root, visited, order);
    
    // Reverse the order because we want to go from Output -> Input
    Collections.reverse(order);
    return order;
  }

  private static void visit(Tensor node, Set<Tensor> visited, List<Tensor> order) {
    if (node == null || visited.contains(node)) return;
    
    visited.add(node);
    if (node.parents != null) {
      for (Tensor parent : node.parents) {
        visit(parent, visited, order);
      }
    }
    order.add(node);
  }

  public void backward() {
    if (!this.requiresGrad) {
      throw new IllegalStateException("Called backward on a tensor that doesn't require gradients.");
    } 

    if (this.parents == null && this.gradFunc == null) {
      throw new IllegalStateException(
        "This tensor is the result of a non-differentiable operation."
      );
    }

    this.grad = Tensor.onesLike(this);

    List<Tensor> order = buildGraph(this);

    for (Tensor node : order) {
      if (node.gradFunc != null) {
        node.gradFunc.apply(node.grad);
      }
    }
  }
}
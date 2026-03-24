import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import tensor.core.Engine;
import tensor.core.Generator;
import tensor.core.Traceable;
import tensor.core.Utility;
import tensor.ops.Backwards;
import tensor.ops.Binary;
import tensor.ops.Reduction;
import tensor.ops.Unary;
import tensor.TensorCore;
import tensor.tools.ArrayTools;
import tensor.tools.Statistics;

public class Tensor implements Traceable{
  public final int rank;
  public final int size;
  public final int[] shape;

  private final TensorCore core;
  
  public boolean requiresGrad;
  public Tensor grad;
  public static boolean verbose = true;

  private List<Tensor> parents = new ArrayList<>();
  private Backwards derivative;
  
  public Tensor(double[] data, int... shape) {
    this(new TensorCore(data, shape));
  }

  public Tensor(TensorCore core) {
    this.core = core;
    this.requiresGrad = false;
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
    Tensor out = new Tensor(TensorCore.permute(tensor.core, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      int[] inverse = new int[axes.length];

      for (int i = 0; i < axes.length; i++) {
        inverse[axes[i]] = i;
      }

      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        TensorCore gradInputCore = TensorCore.permute(grad.core, inverse);
        tensor.accumulate(new Tensor(gradInputCore));
      };
    }

    return out;
  }

  public static Tensor reshape(Tensor tensor, int... axes) {
    Tensor out = new Tensor(TensorCore.reshape(tensor.core, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        TensorCore gradInputCore = TensorCore.reshape(grad.core, tensor.shape);
        tensor.accumulate(new Tensor(gradInputCore));
      };
    }

    return out;
  }

  public static Tensor squeeze(Tensor tensor) {
    Tensor out = new Tensor(TensorCore.squeeze(tensor.core));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);
      
      out.derivative = (grad) -> {
        TensorCore reshaped = TensorCore.reshape(grad.core, tensor.shape);
        tensor.accumulate(new Tensor(reshaped));
      };
    }
    
    return out;
  }
  
  public static Tensor unsqueeze(Tensor tensor) {
    Tensor out = new Tensor(TensorCore.unsqueeze(tensor.core));
    out.requiresGrad = tensor.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        TensorCore reshaped = TensorCore.reshape(grad.core, tensor.shape);
        tensor.accumulate(new Tensor(reshaped));
      };
    }

    return out;
  }

  // UNARY

  /**
   * Applies a reduction operation elements specified in the axes of the given tensor. By default, the output tensor cannot automatically compute gradients
   * for this arbitrary reduction operation. reduce() defaults to preserving the dimension(s).
   * 
   * @param tensor the input tensor
   * @param operation the reduction operation to apply
   * @param axes the axes to apply the reduction operation by
   * @return a new Tensor containing the results with gradients set to false
   */
  public static Tensor reduce(Tensor tensor, Reduction operation, int... axes) {
    return new Tensor(TensorCore.reduce(tensor.core, operation, true, axes)).noGrad();
  }

  /**
   * Applies a reduction operation elements specified in the axes of the given tensor. By default, the output tensor cannot automatically compute gradients
   * for this arbitrary reduction operation. The option to collapse the dimension(s) which Tensor preserves by default.
   * 
   * @param tensor the input tensor
   * @param operation the reduction operation to apply
   * @param axes the axes to apply the reduction operation by
   * @param keepDims whether to reduce the dimension(s)
   * @return a new Tensor containing the results with gradients set to false
   */
  public static Tensor reduce(Tensor tensor, Reduction operation, boolean keepDims, int... axes) {
    return new Tensor(TensorCore.reduce(tensor.core, operation, keepDims, axes)).noGrad();
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
    Tensor out = new Tensor(TensorCore.apply(tensor.core, operation));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        TensorCore derivativeCore = TensorCore.apply(tensor.core, derivative);
        TensorCore gradInputCore = TensorCore.elementwise(grad.core, derivativeCore, (a, b) -> a * b);
        tensor.accumulate(new Tensor(gradInputCore));
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
  public static Tensor elementwise(Tensor tensorA, Tensor tensorB, Binary op, Binary dA, Binary dB) {
    if (tensorA.rank != tensorB.rank) {
      throw new IllegalArgumentException("Mismatching rank for binary elementwise operation. Got Tensors of rank " + tensorA.rank + " and " + tensorB.rank);
    }

    int[] broadcastShapeTarget = Utility.broadcastedShape(tensorA.shape, tensorB.shape);

    TensorCore coreA = tensorA.core.broadcast(broadcastShapeTarget);
    TensorCore coreB = tensorB.core.broadcast(broadcastShapeTarget);
  
    Tensor out = new Tensor(TensorCore.elementwise(coreA, coreB, op));

    out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensorA, tensorB);

      out.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          TensorCore dACore = TensorCore.elementwise(coreA, coreB, dA);
          TensorCore gradA  = TensorCore.elementwise(grad.core, dACore, (x, y) -> x * y);
          
          // CRITICAL: If tensorA was smaller than the output, sum the gradient back
          double[] reducedData = Engine.reduceSum(gradA.dump(), broadcastShapeTarget, tensorA.shape);
          tensorA.accumulate(new Tensor(reducedData, tensorA.shape));
        }

        if (tensorB.requiresGrad) {
          TensorCore dBCore = TensorCore.elementwise(coreA, coreB, dB);
          TensorCore gradB  = TensorCore.elementwise(grad.core, dBCore, (x, y) -> x * y);
          
          // CRITICAL: If tensorB was smaller, sum back to original shape
          double[] reducedData = Engine.reduceSum(gradB.dump(), broadcastShapeTarget, tensorB.shape);
          tensorB.accumulate(new Tensor(reducedData, tensorB.shape));
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
  public static Tensor contract(Tensor tensorA, Tensor tensorB, int[] axesA, int[] axesB) {
    Tensor out = new Tensor(TensorCore.contract(tensorA.core, tensorB.core, axesA, axesB));
    out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensorA, tensorB);

      // returns the grad axes of A and B
      int[][] survivors = Utility.getResultAxes(tensorA.shape, tensorB.shape, axesA, axesB);

      out.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          TensorCore gradACore = TensorCore.contract(grad.core, tensorB.core, survivors[1], axesB);
          tensorA.accumulate(new Tensor(gradACore));
        }

        if (tensorB.requiresGrad) {
          TensorCore gradBCore = TensorCore.contract(tensorA.core, grad.core, axesA, survivors[0]);
          tensorB.accumulate(new Tensor(gradBCore));
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

      out.derivative = (grad) -> {
        double[] gradInput = Engine.transformData(grad.dump(), grad.shape, input.shape);
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

      out.derivative = (grad) -> {
        double[] gradExpanded = Engine.transformData(grad.dump(), grad.shape, tensor.shape);
        double[] yExpanded = Engine.transformData(out.core.dump(), out.shape, tensor.shape);

        double[] xData = tensor.core.dump();
        double[] gradInput = new double[tensor.size];

        for (int i = 0; i < gradInput.length; i++) {
          gradInput[i] = gradExpanded[i] * (yExpanded[i] / xData[i]);
        }

        tensor.accumulate(new Tensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  // note: min and max grads are still wrong: they both assume its global when its not guarenteed
  public Tensor max(int... axes) {return max(this, axes);}
  public static Tensor max(Tensor tensor, int... axes) {
    Tensor out = new Tensor(TensorCore.reduce(tensor.core, Statistics::max, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        double[] gradExpanded = Engine.transformData(grad.dump(), grad.shape, tensor.shape);
        double[] yExpanded = Engine.transformData(out.core.dump(), out.shape, tensor.shape);

        double[] xData = tensor.core.dump();
        double[] gradInput = new double[tensor.size];

        int count = 0;
        for (double n: xData) count += (n == yExpanded[0]) ? 1 : 0;
        for (int i = 0; i < gradInput.length; i++) gradInput[i] = (xData[i] == yExpanded[i]) ? gradExpanded[i] / count : 0;

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

      out.derivative = (grad) -> {
        double[] gradExpanded = Engine.transformData(grad.dump(), grad.shape, tensor.shape);
        double[] yExpanded = Engine.transformData(out.core.dump(), out.shape, tensor.shape);

        double[] xData = tensor.core.dump();
        double[] gradInput = new double[tensor.size];

        int count = 0;
        for (double n: xData) count += (n == yExpanded[0]) ? 1 : 0;
        for (int i = 0; i < gradInput.length; i++) gradInput[i] = (xData[i] == yExpanded[i]) ? gradExpanded[i] / count : 0;

        tensor.accumulate(new Tensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  // non-atomic reductions

  public Tensor mean(int... axes) {return mean(this, axes);}
  public static Tensor mean(Tensor tensor, int... axes) {
    Tensor sum = sum(tensor, axes);
    int reductionSize = Utility.sizeOf(Utility.getSubShape(tensor.shape, axes));

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

  public Tensor median(int... axes) {return median(this, axes);}
  public static Tensor median(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("median() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::median, axes)).noGrad();
  }

  public Tensor mode(int... axes) {return mode(this, axes);}
  public static Tensor mode(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("mode() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::mode, axes)).noGrad();
  }

  public Tensor quartile1(int... axes) {return quartile1(this, axes);}
  public static Tensor quartile1(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("quartile1() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::quartile1, axes)).noGrad();
  }

  public Tensor quartile3(int... axes) {return quartile3(this, axes);}
  public static Tensor quartile3(Tensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("quartile3() is not differentiable");
    }

    return new Tensor(TensorCore.reduce(tensor.core, Statistics::quartile3, axes)).noGrad();
  }

  public Tensor iqr(int... axes) {return iqr(this, axes);}
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

  public static Tensor add(Tensor tensorA, Tensor tensorB) {
    return Tensor.elementwise(tensorA, tensorB, (x, y) -> x + y, (x, y) -> 1.0, (x, y) -> 1.0);
  }

  public static Tensor sub(Tensor tensorA, Tensor tensorB) {
    return Tensor.elementwise(tensorA, tensorB, (x, y) -> x - y, (x, y) -> 1.0, (x, y) -> -1.0);
  }

  public static Tensor hadamard(Tensor tensorA, Tensor tensorB) {
    return Tensor.elementwise(tensorA, tensorB, (x, y) -> x * y, (x, y) -> y, (x, y) -> x);
  }

  public static Tensor div(Tensor tensorA, Tensor tensorB) {
    return Tensor.elementwise(tensorA, tensorB, (x, y) -> x / y, (x, y) -> 1.0 / y, (x, y) -> -x / Math.pow(y, 2));
  }

  public static Tensor pow(Tensor tensorA, Tensor tensorB) {
    return Tensor.elementwise(tensorA, tensorB, (x, y) -> Math.pow(x, y), (x, y) -> y * Math.pow(x, y - 1), (x, y) -> Math.pow(x, y) * Math.log(x));
  }

  // function

  public static Tensor rescaleStandard(Tensor tensor){
    double mean = Statistics.mean(tensor.core.dump());
    double stdDev = Statistics.stdev(tensor.core.dump());

    return apply(tensor, n -> (n - mean) / stdDev, n -> 1.0 / stdDev);
  }

  public static Tensor rescaleMinMax(Tensor tensor){
    double max = Statistics.max(tensor.core.dump());
    double min = Statistics.min(tensor.core.dump());

    return apply(tensor, n -> (n - min) / (max - min), n -> 1.0 / (max - min));
  }
  
  public static Tensor rescaleRobust(Tensor tensor){
    double median = Statistics.median(tensor.core.dump());
    double iqr = Statistics.iqr(tensor.core.dump());
    
    return apply(tensor, n -> (n - median) / iqr, n -> 1.0 / iqr);
  }
  
  public static Tensor rescaleMaxAbs(Tensor tensor){
    double maxAbs = Statistics.max(ArrayTools.foreach(tensor.core.dump(), n -> Math.abs(n)));
    
    return apply(tensor, n -> n / maxAbs, n -> 1.0 / maxAbs);
  }

  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  @Override public List<Tensor> getParents() { return Collections.unmodifiableList(this.parents); }
  @Override public int[] getShape() { return this.shape.clone(); }
  @Override public Backwards getGradFunc() { return this.derivative; }
    
  /**
   * Returns a flat double array containing the elements of this tensor in row-major order.
   * 
   * @return a flat array containing the elements of this tensor
   */
  public double[] dump() {
    return this.core.dump();
  }

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
    tensor.derivative = null;

    return tensor;
  }

  public double get(int... indices) {
    return this.core.get(indices);
  }

  @Override
  public String toString() {
    if (this.core.dump() == null || this.shape == null || this.core.dump().length == 0) return "Tensor[null]";

    String prefix = "Tensor" + Arrays.toString(this.shape) + "(\n";
    String content = Utility.print(this.core.dump(), this.shape, this.core.getStrides(), 0, 0, 2);
    String suffix = " grad=" + this.requiresGrad;

    if (verbose) {
      return prefix + content + "\n\n" + suffix + "\n)";
    } else {
      return prefix + content + "\n)";
    }
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
    this.grad = new Tensor(TensorCore.elementwise(this.grad.core, incomingGrad.core, (a, b) -> a + b));
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

    if (this.parents == null && this.derivative == null) {
      throw new IllegalStateException(
        "This tensor is the result of a non-differentiable operation."
      );
    }

    this.grad = Tensor.onesLike(this);

    List<Tensor> order = buildGraph(this);

    for (Tensor node : order) {
      if (node.derivative != null) {
        node.derivative.apply(node.grad);
      }
    }
  }
}
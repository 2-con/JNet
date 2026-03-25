package stats;

import tensor.core.backend.compute.Shaping;
import tensor.core.impl.TensorCore;
import tensor.functional.main.CoreReductionOps;
import tensor.graph.main.BinaryOps;
import tensor.graph.main.ReductionOps;
import tensor.graph.main.UnaryOps;

public class Measure {
  public static TensorCore mean(TensorCore tensor, int... axes) {
    TensorCore sum = ReductionOps.sum(tensor, axes);
    int reductionSize = Statistics.prod(Shaping.getSubShape(tensor.shape, axes));

    return UnaryOps.mul(sum, 1.0 / reductionSize);
  }

  public static TensorCore variance(TensorCore tensor, int... axes) {
    TensorCore mean = mean(tensor, axes);

    TensorCore centered = BinaryOps.sub(tensor, mean);
    TensorCore sq = UnaryOps.pow(centered, 2);

    return mean(sq, axes);
  }

  public static TensorCore stdev(TensorCore tensor, int... axes) {
    return UnaryOps.pow(variance(tensor, axes), 0.5);
  }

  public static TensorCore skew(TensorCore tensor, int... axes) {
    TensorCore mean = mean(tensor, axes);
    TensorCore centered = BinaryOps.sub(tensor, mean);

    TensorCore m3 = mean(UnaryOps.pow(centered, 3), axes);
    TensorCore stdev = stdev(tensor, axes);

    return BinaryOps.div(m3, UnaryOps.pow(stdev, 3));
  }

  public static TensorCore kurtosis(TensorCore tensor, int... axes) {
    TensorCore mean = mean(tensor, axes);
    TensorCore centered = BinaryOps.sub(tensor, mean);

    TensorCore m4 = mean(UnaryOps.pow(centered, 4), axes);
    TensorCore stdev = stdev(tensor, axes);

    return BinaryOps.div(m4, UnaryOps.pow(stdev, 4));
  }

  // non-differentiable

  public static TensorCore range(TensorCore tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("range() is not differentiable");
    }

    return new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::range, axes)).noGrad();
  }

  public static TensorCore median(TensorCore tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("median() is not differentiable");
    }

    return new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::median, axes)).noGrad();
  }

  public static TensorCore mode(TensorCore tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("mode() is not differentiable");
    }

    return new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::mode, axes)).noGrad();
  }

  public static TensorCore quartile1(TensorCore tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("quartile1() is not differentiable");
    }

    return new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::quartile1, axes)).noGrad();
  }

  public static TensorCore quartile3(TensorCore tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("quartile3() is not differentiable");
    }

    return new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::quartile3, axes)).noGrad();
  }

  public static TensorCore iqr(TensorCore tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("iqr() is not differentiable");
    }

    return new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::iqr, axes)).noGrad();
  }
}

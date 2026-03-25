package stats;

import tensor.core.impl.TensorCore;
import tensor.graph.main.BinaryOps;
import tensor.graph.main.ReductionOps;
import tensor.graph.main.UnaryOps;

public class Scaler {
  public static TensorCore rescaleStandard(TensorCore tensor){
    TensorCore mean = Measure.mean(tensor, tensor.allAxes);
    TensorCore stdev = Measure.stdev(tensor, tensor.allAxes);

    return BinaryOps.div(BinaryOps.sub(tensor, mean), stdev);
  }

  public static TensorCore rescaleMinMax(TensorCore tensor){
    TensorCore max = ReductionOps.max(tensor, tensor.allAxes);
    TensorCore min = ReductionOps.min(tensor, tensor.allAxes);

    return BinaryOps.div(BinaryOps.sub(tensor, min), BinaryOps.sub(max, min));
  }
  
  public static TensorCore rescaleRobust(TensorCore tensor){
    TensorCore median = Measure.median(tensor, tensor.allAxes);
    TensorCore iqr = Measure.iqr(tensor, tensor.allAxes);

    return BinaryOps.div(BinaryOps.sub(tensor, median), iqr);
  }
  
  public static TensorCore rescaleMaxAbs(TensorCore tensor){
    TensorCore maxAbs = ReductionOps.max(UnaryOps.apply(tensor, (n) -> Math.abs(n), (n) -> (n > 0) ? 1.0 : -1.0));
    
    return BinaryOps.div(tensor, maxAbs);
  }
}

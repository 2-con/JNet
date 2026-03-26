package com.aufy.jnet.stats.tensor;

import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.graph.main.BinaryOps;
import com.aufy.jnet.tensor.graph.main.ReductionOps;
import com.aufy.jnet.tensor.graph.main.UnaryOps;

public class Rescale {
  public static CoreTensor standard(CoreTensor tensor){
    CoreTensor mean = Measure.mean(tensor, tensor.allAxes);
    CoreTensor stdev = Measure.stdev(tensor, tensor.allAxes);

    return BinaryOps.div(BinaryOps.sub(tensor, mean), stdev);
  }

  public static CoreTensor minMax(CoreTensor tensor){
    CoreTensor max = ReductionOps.max(tensor, tensor.allAxes);
    CoreTensor min = ReductionOps.min(tensor, tensor.allAxes);

    return BinaryOps.div(BinaryOps.sub(tensor, min), BinaryOps.sub(max, min));
  }
  
  public static CoreTensor eobust(CoreTensor tensor){
    CoreTensor median = Measure.median(tensor, tensor.allAxes);
    CoreTensor iqr = Measure.iqr(tensor, tensor.allAxes);

    return BinaryOps.div(BinaryOps.sub(tensor, median), iqr);
  }
  
  public static CoreTensor maxAbs(CoreTensor tensor){
    CoreTensor maxAbs = ReductionOps.max(UnaryOps.apply(tensor, (n) -> Math.abs(n), (n) -> (n > 0) ? 1.0 : -1.0));
    
    return BinaryOps.div(tensor, maxAbs);
  }
}

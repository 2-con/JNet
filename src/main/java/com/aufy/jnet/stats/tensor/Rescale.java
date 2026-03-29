package com.aufy.jnet.stats.tensor;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.tensor.core.backend.compute.Generator;

public class Rescale {
  public static Tensor standard(Tensor tensor){
    Tensor mean = Measure.mean(tensor, Generator.arrange(tensor.rank));
    Tensor stdev = Measure.stdev(tensor, Generator.arrange(tensor.rank));

    return Tensor.div(Tensor.sub(tensor, mean), stdev);
  }

  public static Tensor minMax(Tensor tensor){
    Tensor max = Tensor.max(tensor, Generator.arrange(tensor.rank));
    Tensor min = Tensor.min(tensor, Generator.arrange(tensor.rank));

    return Tensor.div(Tensor.sub(tensor, min), Tensor.sub(max, min));
  }
  
  public static Tensor eobust(Tensor tensor){
    Tensor median = Measure.median(tensor, Generator.arrange(tensor.rank));
    Tensor iqr = Measure.iqr(tensor, Generator.arrange(tensor.rank));

    return Tensor.div(Tensor.sub(tensor, median), iqr);
  }
  
  public static Tensor maxAbs(Tensor tensor){
    Tensor maxAbs = Tensor.max(Tensor.apply(tensor, (n) -> Math.abs(n), (n) -> (n > 0) ? 1.0 : -1.0));
    
    return Tensor.div(tensor, maxAbs);
  }
}

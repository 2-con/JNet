package com.aufy.jnet;
import com.aufy.jnet.tensor.core.impl.CoreTensor;

public class Tensor extends CoreTensor {
  public Tensor(double[] data, int... shape){
    super(data, shape);
  }
  
  public Tensor(CoreTensor tensor) {
    super(tensor.dump(), tensor.shape);
  }
}

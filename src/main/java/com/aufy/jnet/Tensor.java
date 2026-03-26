package com.aufy.jnet;
import com.aufy.jnet.tensor.core.impl.TensorCore;

public class Tensor extends TensorCore {
  public Tensor(double[] data, int... shape){
    super(data, shape);
  }
}

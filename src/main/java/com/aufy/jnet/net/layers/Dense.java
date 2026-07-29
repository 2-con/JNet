package com.aufy.jnet.net.layers;

import java.util.List;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;
import com.aufy.jnet.net.Parameter;
import com.aufy.jnet.statistics.distributions.Gaussian;

public class Dense extends Module {
  
  private Parameter weight;
  private Parameter bias;

  public Dense(int incomingDimensions, int neurons) {

    // temporary; add custom initializers soon
    weight = new Parameter(new Tensor(new Gaussian(0, 1), neurons, incomingDimensions));
    bias = new Parameter(Tensor.zeros(1, neurons));
  }

  @Override
  public Tensor forward(Tensor x) {
    // return Tensor.add(Tensor.matmul(x, weight.transpose()), bias);


    // System.out.println("Dense ========");
    // System.out.println(x);
    // System.out.println(weight);
    // System.out.println(bias);



    return x.matmul(weight.core.transpose()).add(bias.core);
  }

  @Override
  public List<Parameter> parameters() {
    return List.of(weight, bias);
  }
}

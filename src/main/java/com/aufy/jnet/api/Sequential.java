package com.aufy.jnet.api;

import java.util.ArrayList;
import java.util.List;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;
import com.aufy.jnet.net.Parameter;

public class Sequential extends Module {

  private final List<Module> modules;

  public Sequential(Module... modules) {
    this.modules = List.of(modules);
  }

  @Override
  public Tensor forward(Tensor input) {
    Tensor x = input;

    for (Module module : modules) x = module.forward(x);

    return x;
  }

  @Override
  public List<Parameter> parameters() {
    List<Parameter> params = new ArrayList<>();

    for (Module module : modules) params.addAll(module.parameters());

    return params;
  }

  @Override
  public void train() {
    for (Module module : modules) module.train();
  }

  @Override
  public void eval() {
    for (Module module : modules) module.eval();
    
  }

  public Module get(int index) {
    return modules.get(index);
  }

  public int size() {
    return modules.size();
  }
}

package com.aufy.jnet.net.schedulers;

import com.aufy.jnet.net.optimizers.Optimizer;

public abstract class Scheduler {
  /*
  dont worry about this yet, just for future reference
  */

  protected Optimizer optimizer;

  public abstract void step();

}
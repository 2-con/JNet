package com.aufy.jnet.tensor.core.impl;

import java.util.List;

public interface Traceable {
  List<? extends Traceable> getParents();
  int[] getShape();
  Object getGradFunc();
}

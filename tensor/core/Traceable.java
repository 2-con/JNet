package tensor.core;

import java.util.List;
import tensor.ops.Backwards;

public interface Traceable {
  List<? extends Traceable> getParents();
  int[] getShape();
  Backwards getGradFunc();
}

package tensor.ops;

@FunctionalInterface
public interface Backwards {
  void backward(Tensor grad);
}

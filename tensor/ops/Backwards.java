package tensor.ops;

@FunctionalInterface
public interface Backwards {
  void apply(Tensor grad);
}

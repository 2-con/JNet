package com.aufy.jnet.net;

public interface Callback {
  /*
  only for sequential()? idk how to do this yet so these are just some ideas
  */

  default void epochStart() {}

  default void epochEnd() {}

  default void batchStart() {}

  default void batchEnd() {}

}

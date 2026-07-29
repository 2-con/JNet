package com.aufy.jnet.core.backend.exceptions;

public class Message {
  private static String methodString(String operationName) {
    return (operationName == null || operationName.isBlank()) ? "" : "for '" + operationName + "'";
  }

  public static String crash(String responsibility, String messege, String operationName, String cause) {
    return "[" + responsibility.toUpperCase() + " ERROR] " + messege + " " + methodString(operationName) + ": " + cause;
  }

  public static String warning(String responsibility, String messege, String operationName, String cause) {
    return "[WARNING] " + messege + " " + methodString(operationName) + ": " + cause;
  }
}

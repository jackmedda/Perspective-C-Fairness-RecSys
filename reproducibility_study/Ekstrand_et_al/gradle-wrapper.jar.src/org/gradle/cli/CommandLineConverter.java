package org.gradle.cli;

public interface CommandLineConverter<T> {
  T convert(Iterable<String> paramIterable, T paramT) throws CommandLineArgumentException;
  
  T convert(ParsedCommandLine paramParsedCommandLine, T paramT) throws CommandLineArgumentException;
  
  void configure(CommandLineParser paramCommandLineParser);
}
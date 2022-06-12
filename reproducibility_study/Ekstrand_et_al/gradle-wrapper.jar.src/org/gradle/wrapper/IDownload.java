package org.gradle.wrapper;

import java.io.File;
import java.net.URI;

public interface IDownload {
  void download(URI paramURI, File paramFile) throws Exception;
}
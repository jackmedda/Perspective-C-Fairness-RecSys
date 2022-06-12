/*    */ package org.gradle.cli;
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ public class CommandLineArgumentException
/*    */   extends RuntimeException
/*    */ {
/*    */   public CommandLineArgumentException(String message) {
/* 23 */     super(message);
/*    */   }
/*    */   
/*    */   public CommandLineArgumentException(String message, Throwable cause) {
/* 27 */     super(message, cause);
/*    */   }
/*    */ }
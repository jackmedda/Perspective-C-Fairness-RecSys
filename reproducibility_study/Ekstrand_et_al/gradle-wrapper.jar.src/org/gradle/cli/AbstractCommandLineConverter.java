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
/*    */ public abstract class AbstractCommandLineConverter<T>
/*    */   implements CommandLineConverter<T>
/*    */ {
/*    */   public T convert(Iterable<String> args, T target) throws CommandLineArgumentException {
/* 20 */     CommandLineParser parser = new CommandLineParser();
/* 21 */     configure(parser);
/* 22 */     return convert(parser.parse(args), target);
/*    */   }
/*    */ }
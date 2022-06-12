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
/*    */ public class SystemPropertiesCommandLineConverter
/*    */   extends AbstractPropertiesCommandLineConverter
/*    */ {
/*    */   protected String getPropertyOption() {
/* 22 */     return "D";
/*    */   }
/*    */ 
/*    */   
/*    */   protected String getPropertyOptionDetailed() {
/* 27 */     return "system-prop";
/*    */   }
/*    */ 
/*    */   
/*    */   protected String getPropertyOptionDescription() {
/* 32 */     return "Set system property of the JVM (e.g. -Dmyprop=myvalue).";
/*    */   }
/*    */ }
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
/*    */ public class ProjectPropertiesCommandLineConverter
/*    */   extends AbstractPropertiesCommandLineConverter
/*    */ {
/*    */   protected String getPropertyOption() {
/* 23 */     return "P";
/*    */   }
/*    */ 
/*    */   
/*    */   protected String getPropertyOptionDetailed() {
/* 28 */     return "project-prop";
/*    */   }
/*    */ 
/*    */   
/*    */   protected String getPropertyOptionDescription() {
/* 33 */     return "Set project property for the build script (e.g. -Pmyprop=myvalue).";
/*    */   }
/*    */ }
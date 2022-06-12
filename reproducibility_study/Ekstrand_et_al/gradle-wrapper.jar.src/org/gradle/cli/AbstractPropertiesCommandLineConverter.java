/*    */ package org.gradle.cli;
/*    */ 
/*    */ import java.util.Map;
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
/*    */ 
/*    */ 
/*    */ public abstract class AbstractPropertiesCommandLineConverter
/*    */   extends AbstractCommandLineConverter<Map<String, String>>
/*    */ {
/*    */   public void configure(CommandLineParser parser) {
/* 27 */     CommandLineOption option = parser.option(new String[] { getPropertyOption(), getPropertyOptionDetailed() });
/* 28 */     option = option.hasArguments();
/* 29 */     option.hasDescription(getPropertyOptionDescription());
/*    */   }
/*    */   
/*    */   public Map<String, String> convert(ParsedCommandLine options, Map<String, String> properties) throws CommandLineArgumentException {
/* 33 */     for (String keyValueExpression : options.option(getPropertyOption()).getValues()) {
/* 34 */       int pos = keyValueExpression.indexOf("=");
/* 35 */       if (pos < 0) {
/* 36 */         properties.put(keyValueExpression, ""); continue;
/*    */       } 
/* 38 */       properties.put(keyValueExpression.substring(0, pos), keyValueExpression.substring(pos + 1));
/*    */     } 
/*    */     
/* 41 */     return properties;
/*    */   }
/*    */   
/*    */   protected abstract String getPropertyOption();
/*    */   
/*    */   protected abstract String getPropertyOptionDetailed();
/*    */   
/*    */   protected abstract String getPropertyOptionDescription();
/*    */ }
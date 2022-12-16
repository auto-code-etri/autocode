# autocode

Research on technology that automatically generates high-quality source code from requirements written in natural language, execution examples, or partially written source code.

>-  Automatic source code generation technology that combines new and existing techniques such as machine learning (language model), program synthesis, and software engineering.

![image](./overview_autocode.png)

# Requirements of Source Code Infrastructure

>- Multiple source code collection systems provided by multiple organizations must be provided. 
   Currently, Sungkyunkwan University and Suresoft company are in charge of collecting source codes, and the relevant institutions must collect and store data sets according to some guidelines.

>- A system that individually processes the collected source codes must be provided. 
   In addition, source code and natural language should be processed so that they can be easily and immediately used in a third form such as AST or PDG, 
   which can structurally reflect the control and data flow of the program, rather than being stored as they are.

>- It has a data set in which the collected source code and natural language are stored in the form of a database, 
   and a system for using the source code data set that can be extracted from outside using a specific API must be provided.

>- A inter-conversion system between NL and PL must be provided for automatic generation of ML-based source code

>- Automatic generation system of ML-based source code must provide multiple most appropriate source codes

>- When a user makes a query using NL, a function that converts the automatically generated source code to suit the user's environment must be provided.

>- The automatic source code generation system must be provided in the API form so that third parties can easily use it from the outside.

# AI-Bank Statement Analysis Tool for Auditors

### Introduction
#### Business/ Use Case
Auditors frequently receive numerous bank statements in PDF format for financial audits, compliance reviews, and fraud detection. This tool is designed to assist auditors in analyzing bank statements efficiently by automating the extraction, verification, and analysis of financial transaction data.

This project uses LLM (Large Language Model) technology with RAG (Retrieval Augmented Generation) techniques to help auditors:
- Extract and verify transaction records from bank statements
- Detect anomalies and potential fraud indicators
- Perform compliance checks against regulatory requirements
- Generate comprehensive audit reports and findings
- Compare transactions across multiple periods
- Identify unusual patterns or discrepancies

The tool converts unstructured bank statement PDFs into structured data, stores records in a vector database for efficient retrieval, and enables natural language queries for audit investigations. 


This project mainly can divide into three main parts: 
1. Data Extraction model for PDF file complex format 
2. Embedding model + Vector Database for Store PDF Retrival document
3. LLM Model + RAG technique from data retrieved from database with natural language queries by user 

### Technology use in this project
1. Unstructure Document Preprocssing
- Because the input document complexity, include table, image (chart), I will use several AI model  like OCR , commputer vision model, Vision transformer , layout transformer, Embedding model to extract and analysis the document content from bank statement.
- Complex layout/Context format Analysis by ML model 
- Level 1 analysis: Document layout Analysis
  - Use Computer vision (object detection) AI model to extract component in document content
    - Custom Train Object ddection Model (YOLO) for Detect/recogize the Document Layout Component
    - Detail of the Custom Train YOLO document layout detection model 
    - see my other github project (yolo-base-doc-layout-detection) :  <https://github.com/johnsonhk88/yolo-base-doc-layout-detection> 
  - then use different AI model analysis and extract different type of components context
  -
- Level 2 each component context 
  - use different AI model for extract and recognize different types of docunment components

- Level 3 High level task analysis
  - use AI model Entities 
  - use AI model Sentiment Analysis
  - use AI model Summarization 

- use advance rule base model or Machine learning  model :
  - group and reorganize the data into a user-friendly format. (no experience to build rule to graoup data)
  - Identify common denominators and create headers for each group. (no experence)
  - Display only the differences between similar items (e.g., window sizes, owners) as line items below each header. 
  - Automate the process using AI, enabling the system to self-learn and understand the data structure.  
  - Extract relevant data from PDFs with different layouts and formats.



2. retrieval augmented generation (RAG) with langChain  
- use Embedding model with VectorDB to Retrieve data values by query
- using training dataset for improvement the Text summaration task for conference speakers
- using Advance RAG technique improve retrieval accuracy (e.g. re-ranking, query extensions, auto-merging)

3. LLM Model / Multi-model 
- try to use different open LLM models / multi-model (e.g. LLama3, gemma 2) , prefer use local open LLM models(planning inference LLM model at offline in local machine)
- LLM model use for user-friendly documentation queries and retrieval information interface with natural language

4. LLM Model evaluation
- use truLens or W&B framework for evaluation and debug LLM performance
- LLM evaluation : Content relevance, Answer Relevance, accuary, recall, precision 

5. AI Agent
- Automated audit workflow with multiple specialized agents:
  - Document extraction agent for PDF processing
  - Compliance verification agent for regulatory checks
  - Anomaly detection agent for fraud indicators
  - Financial analysis agent for transaction verification
  - Report generation agent for audit documentation

6. VectorDB
- Store bank statement data in vector database for efficient retrieval
- Enable semantic search for audit queries
- Find similar transactions and patterns across multiple statements

7. SQL Database
- Store structured transaction records for audit trail
- Query historical transaction data
- Generate comparative reports across audit periods

8. Frontend UI
- First version uses Streamlit for auditor-friendly interface
- Later versions will include full stack with Backend RESTful API
- Features: batch statement upload, audit checklist, anomaly highlighting

### Key Auditing Features
- **Transaction Verification**: Cross-reference and verify all transactions
- **Anomaly Detection**: Identify unusual patterns, duplicate payments, or suspicious activities
- **Compliance Checks**: Verify adherence to regulatory requirements and internal controls
- **Reconciliation**: Compare statements across periods for discrepancies
- **Audit Trail**: Maintain complete record of analysis and findings
- **Natural Language Queries**: Ask questions like "Show all transactions over $10,000" or "Find duplicate payments"



### Installation and Setup
1. use requirements.txt for installation package dependencies
2. you can setup virual environment by venv 
3. add your google api key to .env file  for enviroment variables
4. install pytesseract library for ubuntu linux , please run install-pytesseract-for-linux.sh script file 

### Run Application
1. For Development version:
    go to dev folder run jupyter notebook code for development new model/techique
     
```bash
cd src/dev/
```
2. For Application GUI version: 
    running steamlit run apps.py for develop the application

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import AgentState
from resources import llm, retriever, calculate_premium

# --- LOGGING SETUP ---
logging.basicConfig(
    filename='agent_debug.log', 
    filemode='a', # 'a' for append, 'w' to overwrite each time
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InsuranceAgent")

def intent_router(state: AgentState):
    """
    Classifies user intent: 'SUPPORT' vs 'LEAD_GEN'.
    """
    logger.info("--- Entering Intent Router ---")
    
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Logic: If already collecting info, bypass classification
    current_dialog_state = state.get("dialog_state")
    if current_dialog_state == "collecting_info":
        logger.info(f"Existing dialog state is '{current_dialog_state}'. Bypassing router.")
        return {"dialog_state": "collecting_info"}
    
    logger.info(f"Classifying intent for message: '{last_message[:50]}...'")
    
    system_prompt = """You are an intelligent router. 
    Classify the user's input into:
    1. 'SUPPORT': Questions about coverage, terms, deductibles or just having normal conversation
    2. 'LEAD_GEN': Expressions of interest in buying, getting a quote, or pricing.
    
    Return ONLY the word 'SUPPORT' or 'LEAD_GEN'."""
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    intent = response.content.strip()
    
    logger.info(f"Router classified intent as: {intent}")
        
    if intent == "LEAD_GEN":
        return {"dialog_state": "collecting_info"}
    else:
        return {"dialog_state": "support"}
def support_node(state: AgentState):
    """
    RAG Node: Answers policy questions using the vector store.
    """
    logger.info("--- Entering Support Node (RAG) ---")
    
    query = state["messages"][-1].content
    
    logger.info("Retrieving documents...")
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    
    logger.info(f"Retrieved {len(docs)} documents. Context length: {len(context)} chars")
    
    # 1. STRICT SYSTEM PROMPT
    # We explicitly tell it to refuse answering if info is missing.
    system_prompt = """You are a specialized insurance support assistant.
    Your strictly limited role is to answer questions based ONLY on the provided context.
    
    RULES:
    1. You must answer using ONLY the information found in the 'Context' section below.
    2. Do NOT use any outside knowledge, prior training data, or common sense.
    3. If the answer is not explicitly written in the context, you must say: "I'm sorry, I cannot find that information in the policy documents."
    4. Do not make up facts or attempt to be helpful by guessing.
    5. YOu can answer the users greeting, but do not answer further. just say i can help with answering questions about policy.
    """
    
    # 2. USER INPUT WITH CONTEXT
    user_content = f"""
    Context:
    {context}
    
    User Question: 
    {query}
    """
    
    # 3. INVOKE WITH MESSAGE LIST
    # Chat models follow instructions better when separated into System/Human messages
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ])
    
    logger.info("Generated RAG response.")
    
    return {"messages": [response]}
def lead_qualifier_node(state: AgentState):
    """
    State Machine: Identifies missing data, validates inputs, and asks the next question.
    """
    logger.info("--- Entering Lead Qualifier Node ---")
    
    user_info = state.get("user_info", {"age": None, "location": None, "income": None})
    last_message = state["messages"][-1]
    
    # Flag to track if the user just provided an invalid age
    age_input_error = False

    # 1. EXTRACT DATA from previous user answer
    if isinstance(last_message, HumanMessage):
        logger.info("Attempting to extract info from user input...")
        
        extraction_prompt = f"""
        Extract these fields if present: age, location, income.
        Age and Income are both always supposed to be integers
        Current info: {user_info}
        User input: "{last_message.content}"
        Return JSON with keys 'age', 'location', 'income'. Keep missing values null.
        """
        
        raw_extract = llm.invoke(extraction_prompt).content
        
        try:
            cleaned_json = raw_extract.replace("```json", "").replace("```", "").strip()
            extracted_data = json.loads(cleaned_json)
            
            logger.info(f"LLM Extracted: {extracted_data}")
            
            # --- AGE VALIDATION LOGIC ---
            if "age" in extracted_data and extracted_data["age"] is not None:
                try:
                    age_val = int(extracted_data["age"])
                    # Check constraints
                    if age_val < 0 or age_val > 110:
                        logger.warning(f"Validation Failed: Age {age_val} is out of bounds.")
                        age_input_error = True
                        # Remove the invalid age so it is NOT saved to user_info
                        del extracted_data["age"]
                    else:
                        logger.info(f"Validation Passed: Age {age_val} is valid.")
                except ValueError:
                    # If LLM returned a non-integer string for age
                    logger.warning("Validation Failed: Age is not a valid integer.")
                    age_input_error = True
                    del extracted_data["age"]
            # ---------------------------

            # Merge valid data
            updated = False
            for key, val in extracted_data.items():
                if val is not None and user_info.get(key) is None:
                    user_info[key] = val
                    updated = True
            
            if updated: logger.info(f"Updated User Info: {user_info}")

        except json.JSONDecodeError:
            logger.error("JSON Parsing failed during extraction.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

    # 2. CHECK MISSING FIELDS
    missing = []
    if not user_info.get("age"): missing.append("age")
    elif not user_info.get("location"): missing.append("location")
    elif not user_info.get("income"): missing.append("income")

    logger.info(f"Missing fields: {missing}")

    # 3. DETERMINE RESPONSE
    if missing:
        next_field = missing[0]
        
        # SPECIAL CASE: If age is missing AND we just detected an error, show specific error message
        if next_field == "age" and age_input_error:
            logger.info("Asking for age again due to validation error.")
            return {
                "messages": [AIMessage(content="The age you entered seems incorrect. Please provide a valid age between 0 and 110.")],
                "user_info": user_info,
                "dialog_state": "collecting_info"
            }

        # Standard Questions
        question_map = {
            "age": "To generate a quote, I first need to know: How old are you?",
            "location": "Great. What state or city are you located in?",
            "income": "Finally, what is your approximate annual income?"
        }
        
        logger.info(f"Asking user for: {next_field}")
        return {
            "messages": [AIMessage(content=question_map[next_field])],
            "user_info": user_info,
            "dialog_state": "collecting_info"
        }
    
    # 4. COMPLETED
    logger.info("All fields collected. Qualification complete.")
    return {
        "user_info": user_info,
        "dialog_state": "completed"
    }

def agentic_node(state: AgentState):
    """
    Executes the tool call once all data is gathered.
    """
    logger.info("--- Entering Agentic Node (Tool Call) ---")
    
    info = state["user_info"]
    logger.info(f"Calling 'calculate_premium' with data: {info}")
    
    quote_json = calculate_premium.invoke({
        "age": info["age"], 
        "location": info["location"], 
        "income": info["income"]
    })
    
    logger.info(f"Tool returned: {quote_json}")
    
    final_response = f"Thank you! I have generated a quote for you.\n\nJSON Output:\n{quote_json}"
    return {
        "messages": [AIMessage(content=final_response)],
        "dialog_state": "finished"
    }
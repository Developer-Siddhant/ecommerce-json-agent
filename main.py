import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient

# Initialize the FastAPI application
app = FastAPI(title="E-Commerce Review & Complaint Responder API (Llama Edition)")

# SWAPPED TO LLAMA: We use Llama-3.3-70B-Instruct for enterprise-grade reasoning and JSON outputs
client = InferenceClient(model="meta-llama/Llama-3.3-70B-Instruct")

# Define the structure of the incoming request from your client
class ReviewRequest(BaseModel):
    review_text: str

# This system prompt hardcodes the rules so the Llama model acts like a strict data pipeline
SYSTEM_PROMPT = """
You are a strict, automated E-Commerce Customer Support API. 
Your job is to read an angry customer review and output ONLY a valid raw JSON object. 
Do not include conversational filler like "Sure, here is your JSON" or markdown formatting. Do not wrap the output in ```json blocks.

The output must exactly follow this schema structure:
{
  "customer_sentiment": "String (Angry, Frustrated, Confused, or Neutral)",
  "core_issue": "String (Summarize the main problem in 5 words or less)",
  "suggested_action": "String (Refund, Replacement, or Apology Only)",
  "automated_reply_english": "String (A highly professional, deeply empathetic response under 60 words addressing the core issue directly)",
  "automated_reply_native": "String (The exact same response translated perfectly into the customer's native language if the review wasn't originally in English. If it was in English, repeat the English response here.)"
}
"""

@app.post("/process-review")
async def process_review(request: ReviewRequest):
    try:
        # Format the prompt structure for the open-source Llama model
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Review to process: {request.review_text}"}
        ]
        
        # Call the Hugging Face serverless engine running Llama
        response = client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.1 # Crucial for keeping Llama strictly following the JSON format
        )
        
        # Extract the raw string output from the model
        raw_json_output = response.choices.message.content
        
        # Clean up any accidental leading/trailing spaces
        clean_output = raw_json_output.strip()
        
        # Return the clean structured data straight back to your client
        return {"status": "success", "data": clean_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

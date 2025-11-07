import boto3        # AWS SDK for Python - allows us to interact with AWS services
import json         # For handling JSON data
import subprocess   # For running system commands like curl
import time         # For adding delays and timing operations
from datetime import datetime  # For timestamps and date operations

class Agent():
    def __init__(self):
        """
        Initialize the Agent with a system prompt.s
        """
        self.query_slicing_system_msg = {
            "role": "system",
            "content": """
            You are a smart query parser for a multi-modal product search system.
            Your job is to split any incoming query into two precise outputs: (1) star/rating/price filters and (2) product search terms.

            Format your response as:
            output_1: Extract only information related to star ratings, customer reviews, or price (e.g., "stars above 4", "price under 50"). Summarize clearly and concisely.
            output_2: Extract the remaining product-related details (such as item type, material, finish, color, style, etc.) that are helpful for visual or embedding-based similarity search.
                Remove any duplicates or redundant phrases, and return the product name first with clean, lowercase, comma-separated keywords suitable for retrieval.
            """
        }

        self.recommendation_system_msg = {
            "role": "system",
            "content": """
            You are an intelligent, frinedly, and energentic senior shopping assistant. A user is trying to find the right product based on their preferences.
            You will get the user query, user preferences, and the products retrieved by existing system based on visual and textual similarity to the user's preferences.
            Your goal is to provide the best recommendation, referencing following instructions.

            Instructions:
            1. Recommend 1-3 products from the retrieved products that best match the user's preferences. Prioritize products that match the user’s specific product description (e.g., design, style, material, target audience) in the query_text first.
            Only consider price and rating *after* verifying the product is relevant. Only recommend products provided from the "Retrieved Products" input.
            2. Only mention recommended products in your response. Do not mention products that were not selected.
            3. For each recommended product, provide:
            - Product Name
            - A short summary
            - Price
            - Rating
            - Pros and cons (if relevant)
            - Image URL (include as a standalone link to help user preview visually)
            - Product URL if available (optional, include only if exists)
            4. Be clear, friendly, energentic. Format recommendations in an easy-to-read way.
            5. Do not mention product numbers or internal labels (e.g., "Product 3").
            6. Provide a short final recommendation summary.

            Now generate your recommendation:
        """
        }
        # Create a connection to Amazon Bedrock service
        # Bedrock is AWS's service for accessing AI models like Claude
        self.agent = boto3.client(
            service_name='bedrock-runtime',  # Specify we want the runtime version for making AI calls
            region_name='us-west-2'          # AWS region - using us-west-2 as specified
        )
    
    def filter_request(self, query_text):
        """
        Send a prompt to the agent and get a response.
        
        Args:
            prompt (str): The user's prompt or question.
        
        Returns:
            str: The agent's response.
        """
        # Add the user's prompt to the message history
        messages = [
            self.query_slicing_system_msg,
            {"role": "user", "content": f'Query:\n"{query_text}"'}
        ]
        
        
        # Send the entire message history to Claude and get a response
        response_flag, content = self.call_claude_sonnet(json.dumps(messages))
        # response format = (True, 'output_1: None\n\noutput_2: shirt, red, clothing, apparel')
        lines = content.strip().split('\n')
        output_1 = next((line.replace("output_1:", "").strip() for line in lines if "output_1" in line.lower()), "")
        output_2 = next((line.replace("output_2:", "").strip() for line in lines if "output_2" in line.lower()), "")
        print(f"Output 1 = {output_1}, Output 2 = {output_2}")

        return output_1, output_2

    def call_claude_sonnet(self,messages):
        """
        This function sends a prompt to Claude 4.0 Sonnet and gets a response.
        This is the "brain" of our agent - where the AI thinking happens.
        
        Args:
            messages (str): The question or instruction we want to send to Claude
        
        Returns:
            tuple: (success: bool, response: str) - success status and Claude's response or error message
        """
        
        
        try:
            # Send our prompt to Claude and get a response
            response = self.agent.converse(
                # Specify which version of Claude we want to use
                modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',  # Claude 4.0 Sonnet
                
                # Format our message - Claude expects messages in a specific structure
                messages=[
                    {
                        "role": "user",                    # We are the user asking a question
                        "content": [{"text": messages}]      # Our actual question/prompt
                    }
                ],
                
                # Configure how Claude should respond
                inferenceConfig={
                    "maxTokens": 2000,    # Maximum length of response (tokens ≈ words)
                    "temperature": 0.7,   # Creativity level (0=very focused, 1=very creative)
                    "topP": 0.9          # Another creativity control parameter
                }
            )
            
            # Extract the actual text response from Claude's response structure
            # The response comes nested in a complex structure, so we dig down to get the text
            return True, response['output']['message']['content'][0]['text']
            
        except Exception as e:
            # If something goes wrong, return an error message
            return False, f"Error calling Claude: {str(e)}"
    def genrate_report(self,retrieve_similar_products, query: str, k: int = 5, alpha: float = 0.5, model_choice: str = "Clip-STrans") -> str:
        output_1, output_2 = self.filter_request(query)

        retrieved_docs = retrieve_similar_products(output_2, k=k, alpha=alpha, model_choice=model_choice)

        context = "\n\n".join([
            f"Title: {row['title']}\n"
            f"Description: {row.get('description', '')}\n"
            f"Store: {row.get('store', '')}\n"
            f"Features: {row.get('features', '')}\n"
            f"Details: {row.get('details', '')}\n"
            f"Price: {row.get('price', 'N/A')}\n"
            f"Rating: {row.get('average_rating', 'N/A')}\n"
            f"Image: {row['image']}"
            for _, row in retrieved_docs.iterrows()
        ])

        # print("Retrieved products \n", context)

        messages = [
            self.recommendation_system_msg,
            {"role": "user", "content": f'''
            Query: "{query}"

            User preferences (price/rating filters):
            {output_1}

            Retrieved Products:
            {context}
            '''}
            ]

        response_flag, content = self.call_claude_sonnet(json.dumps(messages))

        return content


if __name__ == "__main__":
    agent = Agent()
    filter_keys,object_deatil=agent.filter_request("I want a red shirt")
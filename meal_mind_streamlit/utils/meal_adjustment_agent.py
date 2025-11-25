import streamlit as st
import json
from langchain_community.chat_models import ChatSnowflakeCortex
from langchain.schema import SystemMessage, HumanMessage
from utils.db import (
    get_daily_meal_id, 
    get_meal_detail_id, 
    get_meal_detail_by_id,
    update_meal_detail, 
    get_all_meal_details_for_day, 
    update_daily_nutrition
)

class MealAdjustmentAgent:
    """Agent for handling meal changes, replacements, and restaurant entries"""

    def __init__(self, session, conn):
        self.session = session
        self.conn = conn
        try:
            self.llm = ChatSnowflakeCortex(
                session=self.session,
                model="llama3.1-70b",
                cortex_search_service="MEAL_MIND"
            )
        except Exception as e:
            st.warning(f"Meal Adjustment Agent LLM init failed: {e}")
            self.llm = None

    def process_request(self, user_input, user_id, date, meal_type):
        """
        Process a user's request to change a meal.
        
        Args:
            user_input: The user's description (e.g., "I ate a burger" or "Give me a pasta recipe")
            user_id: User ID
            date: Date of the meal (YYYY-MM-DD)
            meal_type: breakfast, lunch, dinner, or snacks
            
        Returns:
            Dict containing status and message
        """
        if not self.llm:
            return {"status": "error", "message": "Agent offline"}

        # 1. Analyze Intent and Generate Data
        system_prompt = f"""You are a nutrition assistant. The user wants to update their {meal_type} for {date}.
        
        Determine if this is:
        1. A REPORT of food already eaten (e.g., "I had a burger", "I went to a buffet").
        2. A REQUEST for a new recipe/alternative (e.g., "I want pasta instead", "Give me something else").
        3. An APPEND to the current meal (e.g., "I also had a lemonade", "Add an apple").
        
        Generate a JSON response with the new meal details.
        
        IMPORTANT:
        - If it's a REPORT (restaurant/buffet), estimate nutrition accurately. Recipe instructions can be "Eaten out".
        - If it's a REQUEST, generate a healthy recipe matching the description.
        - If it's an APPEND, generate details ONLY for the NEW item(s) to be added.
        - Return ONLY the JSON.
        """
        
        user_prompt = f"""Analyze the input: "{user_input}"
        
        Format:
        {{
            "intent": "report" or "request" or "append",
            "meal_name": "Name of the meal (or item name if append)",
            "ingredients_with_quantities": [{{"ingredient": "name", "quantity": "amount", "unit": "unit"}}],
            "nutrition": {{
                "calories": 0,
                "protein_g": 0,
                "carbohydrates_g": 0,
                "fat_g": 0,
                "fiber_g": 0
            }},
            "recipe": {{
                "instructions": ["step 1", "step 2"],
                "preparation_time": 0,
                "cooking_time": 0,
                "difficulty_level": "easy/medium/hard"
            }}
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Clean JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            meal_data = json.loads(content)
            
            # 2. Update Database
            daily_meal_id = get_daily_meal_id(self.conn, user_id, date)
            if not daily_meal_id:
                return {"status": "error", "message": "No meal plan found for this date."}
            
            detail_id = get_meal_detail_id(self.conn, daily_meal_id, meal_type)
            if not detail_id:
                return {"status": "error", "message": f"No {meal_type} found for this date."}
            
            # Handle APPEND intent
            if meal_data.get('intent') == 'append':
                current_meal = get_meal_detail_by_id(self.conn, detail_id)
                if current_meal:
                    # Merge names
                    meal_data['meal_name'] = f"{current_meal['meal_name']} + {meal_data['meal_name']}"
                    
                    # Merge ingredients
                    current_ingredients = current_meal.get('ingredients_with_quantities', [])
                    new_ingredients = meal_data.get('ingredients_with_quantities', [])
                    meal_data['ingredients_with_quantities'] = current_ingredients + new_ingredients
                    
                    # Merge nutrition
                    current_nutrition = current_meal.get('nutrition', {})
                    new_nutrition = meal_data.get('nutrition', {})
                    
                    merged_nutrition = {}
                    for key in ['calories', 'protein_g', 'carbohydrates_g', 'fat_g', 'fiber_g']:
                        merged_nutrition[key] = current_nutrition.get(key, 0) + new_nutrition.get(key, 0)
                    
                    meal_data['nutrition'] = merged_nutrition
                    
                    # Keep existing recipe/times if not provided or just append note
                    meal_data['recipe'] = current_meal.get('recipe', {})
                    meal_data['preparation_time'] = current_meal.get('preparation_time', 0)
                    meal_data['cooking_time'] = current_meal.get('cooking_time', 0)
            
            # Update the specific meal
            success = update_meal_detail(self.conn, detail_id, meal_data)
            if not success:
                return {"status": "error", "message": "Failed to update meal in database."}
            
            # 3. Recalculate Daily Totals
            all_meals = get_all_meal_details_for_day(self.conn, daily_meal_id)
            
            total_nutrition = {
                "calories": 0,
                "protein": 0,
                "carbohydrates": 0,
                "fat": 0,
                "fiber": 0
            }
            
            for meal in all_meals:
                total_nutrition["calories"] += meal.get("calories", 0)
                total_nutrition["protein"] += meal.get("protein_g", 0)
                total_nutrition["carbohydrates"] += meal.get("carbohydrates_g", 0)
                total_nutrition["fat"] += meal.get("fat_g", 0)
                total_nutrition["fiber"] += meal.get("fiber_g", 0)
            
            # Round values
            for k, v in total_nutrition.items():
                total_nutrition[k] = round(v, 1)
                
            update_daily_nutrition(self.conn, daily_meal_id, total_nutrition)
            
            msg_action = "added to" if meal_data.get('intent') == 'append' else "updated"
            
            return {
                "status": "success", 
                "message": f"Successfully {msg_action} {meal_type}. New item: {meal_data['meal_name']}.",
                "data": meal_data,
                "new_daily_total": total_nutrition
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error processing request: {str(e)}"}
